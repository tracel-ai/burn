use std::path::Path;

use crate::{data::MnistBatcher, model::Model, training::TrainingConfig};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn infer<B: Backend, P: AsRef<Path>>(artifact_dir: P, device: B::Device, item: MnistItem) {
    let config = TrainingConfig::load(artifact_dir.as_ref().join("config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new()
        .load(artifact_dir.as_ref().join("model"), &device)
        .expect("Trained model should exist; run train first");

    let model: Model<B> = config.model.init(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}
