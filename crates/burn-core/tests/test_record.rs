//! Integration test for the non-generic record system (`burn_core::store`).
//!
//! Saves a (nested) module to a burnpack record, loads it back into a freshly initialized
//! module, and checks the parameters round-trip — exercising the public API exactly as a user
//! would (`into_record` / `save` / `load` / `load_record`).

use burn::module::{Module, Param};
use burn::store::ModuleRecord;
use burn::tensor::Tensor;
use burn_core as burn;
use burn_tensor::Device;

#[derive(Module, Debug)]
struct Layer {
    weight: Param<Tensor<2>>,
    bias: Param<Tensor<1>>,
}

impl Layer {
    fn from_values(weight: [[f32; 2]; 2], bias: [f32; 2], device: &Device) -> Self {
        Self {
            weight: Param::from_data(weight, device),
            bias: Param::from_data(bias, device),
        }
    }

    fn zeros(device: &Device) -> Self {
        Self::from_values([[0.0; 2]; 2], [0.0; 2], device)
    }

    fn values(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.weight.val().to_data().to_vec().unwrap(),
            self.bias.val().to_data().to_vec().unwrap(),
        )
    }
}

/// A module with only the `first` layer — its record holds `first.weight` / `first.bias`,
/// a strict subset of [`Mlp`]'s parameters.
#[derive(Module, Debug)]
struct FirstOnly {
    first: Layer,
}

/// Nested module so the record exercises path building (`first.weight`, `second.bias`, ...).
#[derive(Module, Debug)]
struct Mlp {
    first: Layer,
    second: Layer,
}

impl Mlp {
    fn sample(device: &Device) -> Self {
        Self {
            first: Layer::from_values([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0], device),
            second: Layer::from_values([[7.0, 8.0], [9.0, 10.0]], [11.0, 12.0], device),
        }
    }

    fn zeros(device: &Device) -> Self {
        Self {
            first: Layer::zeros(device),
            second: Layer::zeros(device),
        }
    }
}

fn assert_matches_sample(model: &Mlp) {
    let (w1, b1) = model.first.values();
    let (w2, b2) = model.second.values();
    assert_eq!(w1, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(b1, vec![5.0, 6.0]);
    assert_eq!(w2, vec![7.0, 8.0, 9.0, 10.0]);
    assert_eq!(b2, vec![11.0, 12.0]);
}

#[cfg(feature = "std")]
#[test]
fn save_and_load_module_via_file() {
    let device = Default::default();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mlp.bpk");

    // Save a trained module to a record on disk.
    let record = Mlp::sample(&device).into_record();
    assert_eq!(record.len(), 4); // first.weight, first.bias, second.weight, second.bias
    record.save(&path).unwrap();

    // Load the record and apply it onto a freshly initialized (zeroed) module.
    let record = ModuleRecord::load(&path).unwrap();
    let loaded = Mlp::zeros(&device).load_record(record);

    assert_matches_sample(&loaded);
}

#[test]
fn save_and_load_module_via_bytes() {
    let device = Default::default();

    let bytes = Mlp::sample(&device).into_record().into_bytes().unwrap();

    let record = ModuleRecord::from_bytes(bytes).unwrap();
    let loaded = Mlp::zeros(&device).load_record(record);

    assert_matches_sample(&loaded);
}

#[test]
fn missing_parameters_require_allow_partial() {
    let device = Default::default();

    // A record holding only `first.weight` / `first.bias` (a subset of the full Mlp).
    let partial = FirstOnly {
        first: Layer::from_values([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0], &device),
    }
    .into_record();
    let bytes = partial.into_bytes().unwrap();

    // Strict load into the full Mlp fails: `second.*` is missing from the record.
    let strict = ModuleRecord::from_bytes(bytes.clone()).unwrap();
    assert!(Mlp::zeros(&device).try_load_record(strict).is_err());

    // With `allow_partial`, the present params load and the rest keep their init.
    let lenient = ModuleRecord::from_bytes(bytes).unwrap().allow_partial(true);
    let loaded = Mlp::zeros(&device).load_record(lenient);

    let (w1, b1) = loaded.first.values();
    assert_eq!(w1, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(b1, vec![5.0, 6.0]);
    let (w2, b2) = loaded.second.values();
    assert_eq!(w2, vec![0.0, 0.0, 0.0, 0.0]);
    assert_eq!(b2, vec![0.0, 0.0]);
}
