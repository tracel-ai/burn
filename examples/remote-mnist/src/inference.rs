use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, transform::Mapper, vision::MnistDataset},
    },
    prelude::*,
    store::ModuleRecord,
    tensor::Transaction,
};
use mnist::{
    data::{MnistBatcher, MnistMapper},
    model::Model,
    training::ARTIFACT_DIR,
};

const TEST_IMAGES: usize = 1000;
const SHOWN_PREDICTIONS: usize = 10;

/// Classify the first test images on `device` with the model `train` saved.
pub fn infer(device: &Device) {
    let record = ModuleRecord::load(format!("{ARTIFACT_DIR}/model"))
        .expect("A trained model exists; run train first");
    let model = Model::new(device).load_record(record);

    let dataset = MnistDataset::test();
    let mapper = MnistMapper::default();
    let items = (0..TEST_IMAGES)
        .map(|index| mapper.map(&dataset.get(index).expect("MNIST has this many test images")))
        .collect();
    let batch = MnistBatcher::default().batch(items, device);
    let predicted = model.forward(batch.images).argmax(1).flatten::<1>(0, 1);

    let [predicted, expected] = Transaction::default()
        .register(predicted)
        .register(batch.targets)
        .execute()
        .try_into()
        .expect("One result per registered tensor");
    let predicted: Vec<i64> = predicted.iter().collect();
    let expected: Vec<i64> = expected.iter().collect();

    for (predicted, expected) in predicted.iter().zip(&expected).take(SHOWN_PREDICTIONS) {
        println!("predicted {predicted}, expected {expected}");
    }
    let correct = predicted
        .iter()
        .zip(&expected)
        .filter(|(predicted, expected)| predicted == expected)
        .count();
    println!(
        "accuracy on {TEST_IMAGES} test images: {:.2}%",
        100.0 * correct as f64 / TEST_IMAGES as f64
    );
}
