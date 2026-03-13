use burn_core as burn;

use crate::{
    BurnToPyTorchAdapter, ModuleSnapshot, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore,
};
use burn_core::module::{Module, Param};
use burn_nn::{Linear, LinearConfig};
use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

type TestBackend = burn_ndarray::NdArray;

#[derive(Module, Debug)]
struct TestModel<B: Backend> {
    linear: Linear<B>,
    norm_weight: Param<Tensor<B, 1>>,
    norm_bias: Param<Tensor<B, 1>>,
}

impl<B: Backend> TestModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear: LinearConfig::new(4, 2).with_bias(true).init(device),
            norm_weight: Param::from_data([1.0, 1.0], device),
            norm_bias: Param::from_data([0.0, 0.0], device),
        }
    }
}

#[test]
fn pytorch_to_burn_adapter_linear_transpose() {
    let device = Default::default();
    let model = TestModel::<TestBackend>::new(&device);

    // Save with BurnToPyTorch adapter (will transpose linear weights)
    let mut save_store = SafetensorsStore::from_bytes(None).with_to_adapter(BurnToPyTorchAdapter);
    model.save_into(&mut save_store).unwrap();

    // Load with PyTorchToBurn adapter (will transpose back)
    let mut load_store = SafetensorsStore::from_bytes(None).with_from_adapter(PyTorchToBurnAdapter);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        p.set_data(p_save.data().unwrap().as_ref().clone());
    }

    let mut model2 = TestModel::<TestBackend>::new(&device);
    let result = model2.load_from(&mut load_store).unwrap();

    // Should successfully load all tensors
    assert!(!result.applied.is_empty());

    // Verify the linear weights are the same after round-trip
    let weight1 = model.linear.weight.val().to_data();
    let weight2 = model2.linear.weight.val().to_data();

    assert_eq!(weight1.shape, weight2.shape);
    let data1 = weight1.to_vec::<f32>().unwrap();
    let data2 = weight2.to_vec::<f32>().unwrap();

    for (a, b) in data1.iter().zip(data2.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Weights differ after adapter round-trip"
        );
    }
}

#[test]
fn pytorch_to_burn_adapter_norm_rename() {
    let device = Default::default();

    // Create a model with norm-like naming
    #[derive(Module, Debug)]
    struct NormModel<B: Backend> {
        norm_gamma: Param<Tensor<B, 1>>,
        norm_beta: Param<Tensor<B, 1>>,
    }

    impl<B: Backend> NormModel<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                norm_gamma: Param::from_data([1.0, 2.0, 3.0], device),
                norm_beta: Param::from_data([0.1, 0.2, 0.3], device),
            }
        }
    }

    let model = NormModel::<TestBackend>::new(&device);

    // Save with BurnToPyTorch adapter (will rename gamma->weight, beta->bias)
    let mut save_store = SafetensorsStore::from_bytes(None).with_to_adapter(BurnToPyTorchAdapter);
    model.save_into(&mut save_store).unwrap();

    // The saved data should have PyTorch naming convention
    // We can't directly verify the internal names, but we can verify round-trip works

    // Load with PyTorchToBurn adapter (will rename weight->gamma, bias->beta)
    let mut load_store = SafetensorsStore::from_bytes(None).with_from_adapter(PyTorchToBurnAdapter);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        p.set_data(p_save.data().unwrap().as_ref().clone());
    }

    let mut model2 = NormModel::<TestBackend>::new(&device);
    let result = model2.load_from(&mut load_store).unwrap();

    // Should load successfully
    assert!(!result.applied.is_empty());

    // Verify data is preserved
    let gamma1 = model.norm_gamma.val().to_data().to_vec::<f32>().unwrap();
    let gamma2 = model2.norm_gamma.val().to_data().to_vec::<f32>().unwrap();
    let beta1 = model.norm_beta.val().to_data().to_vec::<f32>().unwrap();
    let beta2 = model2.norm_beta.val().to_data().to_vec::<f32>().unwrap();

    assert_eq!(gamma1, gamma2);
    assert_eq!(beta1, beta2);
}

#[test]
fn no_adapter_preserves_original() {
    let device = Default::default();
    let model = TestModel::<TestBackend>::new(&device);

    // Save without adapter
    let mut save_store = SafetensorsStore::from_bytes(None);
    model.save_into(&mut save_store).unwrap();

    // Load without adapter
    let mut load_store = SafetensorsStore::from_bytes(None);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        p.set_data(p_save.data().unwrap().as_ref().clone());
    }

    let mut model2 = TestModel::<TestBackend>::new(&device);
    let result = model2.load_from(&mut load_store).unwrap();

    assert!(result.is_success());
    assert!(!result.applied.is_empty());

    // Verify data is exactly the same
    let weight1 = model.linear.weight.val().to_data();
    let weight2 = model2.linear.weight.val().to_data();

    assert_eq!(weight1.shape, weight2.shape);
    assert_eq!(
        weight1.to_vec::<f32>().unwrap(),
        weight2.to_vec::<f32>().unwrap()
    );
}

#[test]
#[cfg(all(feature = "std", target_has_atomic = "ptr"))]
fn adapter_with_pytorch_import() {
    use crate::PyTorchToBurnAdapter;

    let device = Default::default();

    // Reference the safetensors file from burn-store
    let safetensors_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/safetensors-tests/tests/multi_layer/multi_layer.safetensors"
    );

    // Simple test model that matches some of the PyTorch structure
    #[derive(Module, Debug)]
    struct SimpleNet<B: Backend> {
        fc1: Linear<B>,
    }

    impl<B: Backend> SimpleNet<B> {
        fn new(device: &B::Device) -> Self {
            Self {
                fc1: LinearConfig::new(4 * 8 * 8, 16).init(device),
            }
        }
    }

    // Load with PyTorchToBurn adapter
    let mut store = SafetensorsStore::from_file(safetensors_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .validate(false)
        .allow_partial(true);

    let mut model = SimpleNet::<TestBackend>::new(&device);
    let result = model.load_from(&mut store).unwrap();

    // Should load some tensors (fc1 if it exists in the file)
    // This mainly tests that the adapter works with real PyTorch files
    assert!(!result.applied.is_empty() || !result.missing.is_empty());
}

#[test]
fn half_precision_adapter_round_trip() {
    use crate::HalfPrecisionAdapter;
    use burn_tensor::DType;

    let device = Default::default();
    let model = TestModel::<TestBackend>::new(&device);

    // Save with HalfPrecisionAdapter (F32 -> F16)
    let adapter = HalfPrecisionAdapter::new();
    let mut save_store = SafetensorsStore::from_bytes(None).with_to_adapter(adapter.clone());
    model.save_into(&mut save_store).unwrap();

    // Verify Linear tensors are F16, raw params stay F32 (no recognized module type)
    let save_bytes = match &save_store {
        SafetensorsStore::Memory(p) => p.data().unwrap().as_ref().clone(),
        _ => panic!("Expected memory store"),
    };
    let mut inspect_store = SafetensorsStore::from_bytes(Some(save_bytes.clone()));
    let snapshots = inspect_store.get_all_snapshots().unwrap();
    for (name, snapshot) in snapshots.iter() {
        if name.starts_with("linear") {
            assert_eq!(
                snapshot.dtype,
                DType::F16,
                "Linear tensor '{}' should be F16",
                name
            );
        } else {
            assert_eq!(
                snapshot.dtype,
                DType::F32,
                "Raw param '{}' should stay F32",
                name
            );
        }
    }

    // Load back with same adapter (F16 -> F32)
    let mut load_store = SafetensorsStore::from_bytes(Some(save_bytes)).with_from_adapter(adapter);

    let mut model2 = TestModel::<TestBackend>::new(&device);
    let result = model2.load_from(&mut load_store).unwrap();

    assert!(!result.applied.is_empty());

    // Verify values are close (F32 -> F16 -> F32 has rounding)
    let w1 = model.linear.weight.val().to_data().to_vec::<f32>().unwrap();
    let w2 = model2
        .linear
        .weight
        .val()
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    for (a, b) in w1.iter().zip(w2.iter()) {
        assert!(
            (a - b).abs() < 0.01,
            "Weight values differ too much after F16 round-trip: {} vs {}",
            a,
            b
        );
    }
}

#[test]
fn half_precision_adapter_without_module() {
    use crate::HalfPrecisionAdapter;
    use burn_nn::{LayerNorm, LayerNormConfig};
    use burn_tensor::DType;

    #[derive(Module, Debug)]
    struct MixedModel<B: Backend> {
        linear: Linear<B>,
        norm: LayerNorm<B>,
    }

    let device = Default::default();
    let model = MixedModel::<TestBackend> {
        linear: LinearConfig::new(4, 2).with_bias(true).init(&device),
        norm: LayerNormConfig::new(2).init(&device),
    };

    // Save: exclude LayerNorm from half-precision conversion
    let adapter = HalfPrecisionAdapter::new().without_module("LayerNorm");
    let mut save_store = SafetensorsStore::from_bytes(None).with_to_adapter(adapter);
    model.save_into(&mut save_store).unwrap();

    // Verify: Linear tensors are F16, LayerNorm tensors remain F32
    let save_bytes = match &save_store {
        SafetensorsStore::Memory(p) => p.data().unwrap().as_ref().clone(),
        _ => panic!("Expected memory store"),
    };
    let mut inspect_store = SafetensorsStore::from_bytes(Some(save_bytes));
    let snapshots = inspect_store.get_all_snapshots().unwrap();
    for (name, snapshot) in snapshots {
        if name.starts_with("linear") {
            assert_eq!(
                snapshot.dtype,
                DType::F16,
                "Linear tensor '{}' should be F16",
                name
            );
        } else if name.starts_with("norm") {
            assert_eq!(
                snapshot.dtype,
                DType::F32,
                "LayerNorm tensor '{}' should stay F32",
                name
            );
        }
    }
}

#[test]
fn half_precision_adapter_default_converts_layer_norm() {
    use crate::HalfPrecisionAdapter;
    use burn_nn::{LayerNorm, LayerNormConfig};
    use burn_tensor::DType;

    #[derive(Module, Debug)]
    struct NormModel<B: Backend> {
        linear: Linear<B>,
        norm: LayerNorm<B>,
    }

    let device = Default::default();
    let model = NormModel::<TestBackend> {
        linear: LinearConfig::new(4, 2).with_bias(true).init(&device),
        norm: LayerNormConfig::new(2).init(&device),
    };

    // Default adapter converts LayerNorm
    let adapter = HalfPrecisionAdapter::new();
    let mut save_store = SafetensorsStore::from_bytes(None).with_to_adapter(adapter);
    model.save_into(&mut save_store).unwrap();

    let save_bytes = match &save_store {
        SafetensorsStore::Memory(p) => p.data().unwrap().as_ref().clone(),
        _ => panic!("Expected memory store"),
    };
    let mut inspect_store = SafetensorsStore::from_bytes(Some(save_bytes));
    let snapshots = inspect_store.get_all_snapshots().unwrap();
    for (name, snapshot) in snapshots {
        assert_eq!(
            snapshot.dtype,
            DType::F16,
            "All tensors should be F16 by default, but '{}' is {:?}",
            name,
            snapshot.dtype
        );
    }
}
