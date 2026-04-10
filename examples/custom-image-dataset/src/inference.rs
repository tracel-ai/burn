use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::vision::{Annotation, ImageDatasetItem},
    },
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

use crate::{data::ClassificationBatcher, model::Cnn};

const NUM_CLASSES: u8 = 10;

pub fn infer(artifact_dir: &str, device: Device, item: ImageDatasetItem) {
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = Cnn::new(NUM_CLASSES.into(), &device).load_record(record);

    let mut label = 0;
    if let Annotation::Label(category) = item.annotation {
        label = category;
    };
    let batcher = ClassificationBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    println!("Predicted {predicted} Expected {label:?}");
}
