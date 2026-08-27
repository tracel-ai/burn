//! End-to-end coverage of `BurnpackStore` in file mode.
//!
//! The unit tests reach the writer through hand-built tensors; these go through the API a
//! user actually calls, so they cover the store's own layers (auto-extension, the `overwrite`
//! guard) sitting on top of the atomic write, and the adapter path that produces tensors
//! whose declared metadata and eventual bytes come from different code.

#![cfg(feature = "std")]

// The `Module` derive expands to `::burn::...` paths.
use burn_core as burn;

use burn_core::module::Module;
use burn_core::tensor::{DType, Device};
use burn_nn::{Linear, LinearConfig};
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};

#[derive(Module, Debug)]
struct TestModel {
    linear1: Linear,
    linear2: Linear,
}

impl TestModel {
    fn new(device: &Device) -> Self {
        Self {
            linear1: LinearConfig::new(8, 16).init(device),
            linear2: LinearConfig::new(16, 4).init(device),
        }
    }
}

/// The first weight, as a flat vector, for comparing a model against its round trip.
///
/// Converts to f32 rather than reading the raw dtype: a model loaded from a half-precision
/// save holds F16 params, and the point of comparison is the values, not the storage.
fn first_weight(model: &TestModel) -> Vec<f32> {
    model
        .linear1
        .weight
        .val()
        .to_data()
        .convert_dtype(DType::F32)
        .try_to_vec()
        .unwrap()
}

#[test]
fn file_mode_round_trips_a_module() {
    let device = Device::default();
    let dir = tempfile::tempdir().unwrap();
    // Extensionless on purpose: the store appends `.bpk` (auto-extension), and save and load
    // must agree on the resolved path or the round trip breaks confusingly.
    let path = dir.path().join("model");

    let model = TestModel::new(&device);
    let mut store = BurnpackStore::from_file(&path);
    model.save_into(&mut store).unwrap();

    assert!(
        dir.path().join("model.bpk").exists(),
        "auto-extension should have appended .bpk"
    );

    let mut loaded = TestModel::new(&device);
    let mut store = BurnpackStore::from_file(&path);
    assert!(loaded.load_from(&mut store).unwrap().is_success());

    assert_eq!(first_weight(&loaded), first_weight(&model));
}

#[test]
fn file_mode_can_preserve_an_extensionless_path() {
    let device = Device::default();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model");

    let model = TestModel::new(&device);
    model
        .save_into(&mut BurnpackStore::from_file(&path).auto_extension(false))
        .unwrap();

    assert!(path.exists(), "the exact requested path should be used");
    assert!(
        !dir.path().join("model.bpk").exists(),
        "the writer must not re-append the extension"
    );

    let original = std::fs::read(&path).unwrap();
    assert!(
        TestModel::new(&device)
            .save_into(&mut BurnpackStore::from_file(&path).auto_extension(false))
            .is_err(),
        "overwrite protection should check the exact path"
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);

    let mut loaded = TestModel::new(&device);
    loaded
        .load_from(&mut BurnpackStore::from_file(&path).auto_extension(false))
        .unwrap();
    assert_eq!(first_weight(&loaded), first_weight(&model));
}

#[test]
fn disabled_extension_does_not_fall_back_when_loading() {
    let device = Device::default();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model");

    TestModel::new(&device)
        .save_into(&mut BurnpackStore::from_file(&path))
        .unwrap();
    assert!(path.with_extension("bpk").exists());
    assert!(!path.exists());

    let mut loaded = TestModel::new(&device);
    assert!(
        loaded
            .load_from(&mut BurnpackStore::from_file(&path).auto_extension(false))
            .is_err(),
        "loading with auto-extension disabled should use the exact path"
    );
}

/// `overwrite` is the store's own policy, layered above a writer that always replaces. Pin
/// both directions, including that a refused save leaves the previous container alone.
#[test]
fn overwrite_guards_an_existing_file() {
    let device = Device::default();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.bpk");

    let first = TestModel::new(&device);
    first
        .save_into(&mut BurnpackStore::from_file(&path))
        .unwrap();
    let original = std::fs::read(&path).unwrap();

    // Second model, same path: refused by default and the file is untouched.
    let second = TestModel::new(&device);
    assert!(
        second
            .save_into(&mut BurnpackStore::from_file(&path))
            .is_err()
    );
    assert_eq!(std::fs::read(&path).unwrap(), original);

    // Opting in replaces it, and the replacement is what loads back.
    second
        .save_into(&mut BurnpackStore::from_file(&path).overwrite(true))
        .unwrap();

    let mut loaded = TestModel::new(&device);
    loaded
        .load_from(&mut BurnpackStore::from_file(&path))
        .unwrap();
    assert_eq!(first_weight(&loaded), first_weight(&second));
    assert_ne!(first_weight(&loaded), first_weight(&first));
}

/// Adapters rebuild a tensor with a declared dtype while the bytes are produced later,
/// inside the closure. That is the one place in the save path where `byte_len` and
/// `into_bytes` are computed by different code, so the writer's length check is load-bearing.
#[test]
fn saving_through_an_adapter_round_trips() {
    let device = Device::default();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("half.bpk");

    let model = TestModel::new(&device);
    let mut store = BurnpackStore::from_file(&path).with_to_adapter(HalfPrecisionAdapter::new());
    model.save_into(&mut store).unwrap();

    let mut loaded = TestModel::new(&device);
    let mut store = BurnpackStore::from_file(&path);
    assert!(loaded.load_from(&mut store).unwrap().is_success());

    // F16 is lossy, so compare approximately rather than for equality.
    for (got, want) in first_weight(&loaded)
        .iter()
        .zip(first_weight(&model).iter())
    {
        assert!((got - want).abs() < 1e-2, "{got} vs {want}");
    }
}
