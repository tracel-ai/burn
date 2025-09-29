use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    module::Module,
    record::{NoStdTrainingRecorder, Recorder},
    tensor::backend::Backend,
};
use rgb::RGB8;
use textplots::{Chart, ColorPlot, Shape};

use crate::{
    dataset::{HousingBatcher, HousingDataset, HousingDistrictItem},
    model::{RegressionModelConfig, RegressionModelRecord},
};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device) {
    let record: RegressionModelRecord<B> = NoStdTrainingRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = RegressionModelConfig::new()
        .init(&device)
        .load_record(record);

    // Use a sample of 1000 items from the test split
    let dataset = HousingDataset::test();
    let items: Vec<HousingDistrictItem> = dataset.iter().take(1000).collect();

    let batcher = HousingBatcher::new(device.clone());
    let batch = batcher.batch(items.clone(), &device);
    let predicted = model.forward(batch.inputs);
    let targets = batch.targets;

    // Display the predicted vs expected values
    let predicted = predicted.squeeze_dim::<1>(1).into_data();
    let expected = targets.into_data();

    let points = predicted
        .iter::<f32>()
        .zip(expected.iter::<f32>())
        .collect::<Vec<_>>();

    println!("Predicted vs. Expected Median House Value (in 100,000$)");
    Chart::new_with_y_range(120, 60, 0., 5., 0., 5.)
        .linecolorplot(
            &Shape::Points(&points),
            RGB8 {
                r: 255,
                g: 85,
                b: 85,
            },
        )
        .display();

    // Print a single numeric value as an example
    println!("Predicted {} Expected {}", points[0].0, points[0].1);
}
