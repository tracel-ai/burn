use burn::{
    module::parallel::{DistributedLayeredModel, LayerPlacement},
    prelude::*,
    store::{BurnpackStore, ModuleSnapshot},
    tensor::{Bytes, Distribution, Tolerance},
};

use crate::{
    layered::LayeredModel,
    model::{Model, ModelConfig},
};

/// Split a model trained on one device across `devices`, load its checkpoint straight onto them,
/// and check the split against the single-device model.
pub fn infer(devices: Vec<Device>) {
    assert!(!devices.is_empty(), "no device to place the model on");
    let config = ModelConfig::new();

    // An initialized model stands in for a trained one, and its checkpoint for one read from a file.
    let trained = config.init(&devices[0]);
    let checkpoint = save_checkpoint(&trained);

    let model = load_split(&config, &devices, checkpoint);
    model.print_placement(model.placement());

    let features = Tensor::random(
        [64, config.features],
        Distribution::Default,
        &model.placement().input,
    );
    assert_matches_single_device(&model, &trained, features);
}

/// Split before the weights exist, so each one loads straight onto its own device.
fn load_split(
    config: &ModelConfig,
    devices: &[Device],
    checkpoint: Bytes,
) -> DistributedLayeredModel<LayeredModel> {
    let placement = LayerPlacement::even(devices, config.blocks);
    let model = LayeredModel::new(config, &placement);
    let mut model = DistributedLayeredModel::new(model, &placement);

    let mut store =
        LayeredModel::remap_from_single_device(BurnpackStore::from_bytes(Some(checkpoint)));
    model
        .load_from(&mut store)
        .expect("the checkpoint should cover every layer");
    model
}

fn save_checkpoint(model: &Model) -> Bytes {
    let mut store = BurnpackStore::from_bytes(None);
    model
        .save_into(&mut store)
        .expect("the checkpoint should save");
    store.get_bytes().expect("the checkpoint was saved")
}

fn assert_matches_single_device(
    split: &DistributedLayeredModel<LayeredModel>,
    single: &Model,
    features: Tensor<2>,
) {
    let predictions = split.forward(features.clone());
    println!(
        "predictions {:?} on {:?}",
        predictions.dims(),
        predictions.device()
    );

    let expected = single.forward(features);
    predictions
        .to_device(&expected.device())
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    println!("the predictions match the single-device model");
}
