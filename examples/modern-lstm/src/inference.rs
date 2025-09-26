use crate::{
    dataset::{
        NOISE_LEVEL, NUM_SEQUENCES, SEQ_LENGTH, SequenceBatcher, SequenceDataset,
        SequenceDatasetItem,
    },
    model::LstmNetwork,
    training::TrainingConfig,
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use polars::prelude::*;

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    // Loading model
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model: LstmNetwork<B> = config.model.init(&device).load_record(record);

    let dataset = SequenceDataset::new(NUM_SEQUENCES / 5, SEQ_LENGTH, NOISE_LEVEL);
    let items: Vec<SequenceDatasetItem> = dataset.iter().collect();

    let batcher = SequenceBatcher::default();
    // Put all items in one batch
    let batch = batcher.batch(items, &device);
    let predicted = model.forward(batch.sequences, None);
    let targets = batch.targets;

    let predicted = predicted.squeeze_dim::<1>(1).into_data();
    let expected = targets.squeeze_dim::<1>(1).into_data();

    // Display the predicted vs expected values
    let results = df![
        "predicted" => &predicted.to_vec::<f32>().unwrap(),
        "expected" => &expected.to_vec::<f32>().unwrap(),
    ]
    .unwrap();
    println!("{}", &results.head(Some(10)));
}
