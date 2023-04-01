use std::sync::Arc;

use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    module::{Module, State},
    record::{DefaultRecordSettings, Record},
    tensor::{backend::Backend, f16},
};

use crate::{
    data::{BertCasedTokenizer, TextClassificationBatcher, TextClassificationDataset, Tokenizer},
    model::TextClassificationModelConfig,
    training::ExperimentConfig,
};

pub fn infer<B: Backend, D: TextClassificationDataset + 'static>(
    device: B::Device,
    artifact_dir: &str,
    samples: Vec<String>,
) {
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");
    let tokenizer = Arc::new(BertCasedTokenizer::default());
    let n_classes = D::num_classes();
    let batcher = Arc::new(TextClassificationBatcher::<B>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    ));

    println!("Creating model ...");
    let model = TextClassificationModelConfig::new(
        config.transformer,
        n_classes,
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .init::<B>();

    println!("Loading weights ...");
    let state: State<f16> =
        Record::load::<DefaultRecordSettings>(format!("{artifact_dir}/model").into())
            .expect("Trained model weights");
    let model = model.load(&state.convert()).expect("Can load weights");
    let model = model.to_device(&device);

    println!("Running inference ...");
    let item = batcher.batch(samples.clone());
    let predictions = model.infer(item);

    for (i, text) in samples.into_iter().enumerate() {
        let prediction = predictions.clone().index([i..i + 1]);
        let logits = prediction.to_data();
        let class_index = prediction.argmax(1).into_data().convert::<i32>().value[0];
        let class = D::class_name(class_index as usize);

        println!("\n=== Item {i} ===\n- Text: {text}\n- Logits: {logits}\n- Prediction: {class}\n================");
    }
}
