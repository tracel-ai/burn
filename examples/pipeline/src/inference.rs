use burn::{
    module::pipeline::{Pipeline, PipelinePlacement},
    prelude::*,
    tensor::{Distribution, Tolerance},
};

use crate::model::{Model, ModelConfig};

/// Load a trained record straight onto the stages it was placed on, then check the split
/// against the same weights on one device.
pub fn infer(devices: Vec<Device>) {
    assert!(!devices.is_empty(), "no device to place the model on");

    let config = ModelConfig::new();
    let placement = PipelinePlacement::even(&devices, config.blocks);

    // A trained model's record, which a real program reads from a file with `ModuleRecord::load`.
    // An initialized model stands in for one here, so this runs without training first.
    let trained = config.init(&devices[0]);
    let record = trained.clone().into_record();

    // Placed before the weights exist, so each one loads straight onto its own stage.
    let model = config
        .init(&devices[0])
        .place(&placement)
        .load_record(record);
    model.print_placement(model.placement());

    let features = Tensor::random(
        [64, config.features],
        Distribution::Default,
        &placement.input,
    );
    let predictions = model.forward(features.clone());
    println!(
        "predictions {:?} on {:?}",
        predictions.dims(),
        predictions.device()
    );

    check_against_one_device(trained, &devices[0], features, predictions);
}

/// The split has to compute what the same weights compute on one device.
fn check_against_one_device(
    trained: Model,
    host: &Device,
    features: Tensor<2>,
    predictions: Tensor<2>,
) {
    let blocks = trained.layout().num_blocks();
    let expected = trained
        .place(&PipelinePlacement::even(
            core::slice::from_ref(host),
            blocks,
        ))
        .forward(features.to_device(host));

    predictions
        .to_device(host)
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    println!("the predictions match the model on one device");
}
