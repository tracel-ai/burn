use crate::{data::MNISTBatcher, training::TrainingConfig};
use burn::data::dataset::vision::MNISTItem;
use burn::{
    config::Config,
    data::dataloader::batcher::Batcher,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: MNISTItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init_with::<B>(record);

    let label = item.label;
    let batcher = MNISTBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}
