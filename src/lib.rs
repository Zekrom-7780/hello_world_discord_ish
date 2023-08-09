use discord_flows::{
    model::Message,
    ProvidedBot, Bot,
};
use flowsnet_platform_sdk::logger;
use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
use rust_bert::bert::BertConfig;
use tch::nn::VarStore;

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn run() -> anyhow::Result<()> {
    let discord_token = std::env::var("discord_token").unwrap();
    let bot = ProvidedBot::new(discord_token);
    bot.listen(|msg| handler(&bot, msg)).await;
    Ok(())
}

async fn handler(bot: &ProvidedBot, msg: Message) {
    logger::init();
    let discord = bot.get_client();

    if msg.author.bot {
        log::debug!("ignored bot message");
        return;
    }
    if msg.member.is_some() {
        log::debug!("ignored channel message");
        return;
    }

    let channel_id = msg.channel_id;

    // Process user messages containing questions
    if msg.content.starts_with("!ask") {
        let question = msg.content.trim_start_matches("!ask").trim();
        
        if question.is_empty() {
            return;
        }

        // Initialize the question answering model
        let qa_model = load_qa_model();

        // Use the question answering model to get an answer
        let qa_input = QaInput {
            question: question.to_string(),
            context: "Provide your context here.".to_string(), // You may customize the context based on your use case
            title: None,
        };

        let answer = qa_model.predict(&[qa_input]).pop().unwrap();
        let answer_text = answer.answer;

        let resp = format!(
            "You asked: '{}'\nQuestion Answer: {}\n",
            question,
            answer_text
        );

        discord
            .send_message(channel_id.into(), &serde_json::json!({ "content": resp }))
            .await;
    }
}

fn load_qa_model() -> QuestionAnsweringModel {
    let config = BertConfig::from_pretrained("bert-base-uncased");
    let vs = VarStore::new(tch::Device::Cpu);
    let model = QuestionAnsweringModel::new(&vs.root(), &config);

    model
}
