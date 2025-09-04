use super::*;
use burn_core as burn;
use burn_core::module::{Module, Param};
use burn_core::nn::{Linear, LinearConfig};
use burn_core::persist::{ModulePersist, PathFilter};
use burn_core::tensor::backend::Backend;
use burn_tensor::Tensor;

type TestBackend = burn_ndarray::NdArray;

#[derive(Module, Debug)]
struct ComplexModule<B: Backend> {
    encoder: EncoderModule<B>,
    decoder: DecoderModule<B>,
    layers: Vec<Linear<B>>,
}

#[derive(Module, Debug)]
struct EncoderModule<B: Backend> {
    weight: Param<Tensor<B, 3>>,
    bias: Param<Tensor<B, 1>>,
    norm: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
struct DecoderModule<B: Backend> {
    projection: Linear<B>,
    scale: Param<Tensor<B, 2>>,
}

impl<B: Backend> ComplexModule<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            encoder: EncoderModule {
                weight: Param::from_data(
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                    device,
                ),
                bias: Param::from_data([0.1, 0.2, 0.3], device),
                norm: Param::from_data([1.0, 1.0, 1.0], device),
            },
            decoder: DecoderModule {
                projection: LinearConfig::new(4, 2).with_bias(true).init(device),
                scale: Param::from_data([[0.5, 0.5], [0.5, 0.5]], device),
            },
            layers: vec![
                LinearConfig::new(3, 4).with_bias(false).init(device),
                LinearConfig::new(4, 3).with_bias(true).init(device),
            ],
        }
    }

    fn new_zeros(device: &B::Device) -> Self {
        Self {
            encoder: EncoderModule {
                weight: Param::from_tensor(Tensor::zeros([2, 2, 2], device)),
                bias: Param::from_tensor(Tensor::zeros([3], device)),
                norm: Param::from_tensor(Tensor::zeros([3], device)),
            },
            decoder: DecoderModule {
                projection: LinearConfig::new(4, 2).with_bias(true).init(device),
                scale: Param::from_tensor(Tensor::zeros([2, 2], device)),
            },
            layers: vec![
                LinearConfig::new(3, 4).with_bias(false).init(device),
                LinearConfig::new(4, 3).with_bias(true).init(device),
            ],
        }
    }
}

#[test]
fn test_complex_module_round_trip() {
    let device = Default::default();
    let module1 = ComplexModule::<TestBackend>::new(&device);
    let mut module2 = ComplexModule::<TestBackend>::new_zeros(&device);

    // Save module1 using new persister API
    let mut save_persister = SafetensorsPersisterConfig::new().build_memory();
    module1.collect_to(&mut save_persister).unwrap();

    // Load into module2
    let mut load_persister = SafetensorsPersisterConfig::new().build_memory();
    if let SafetensorsPersister::Memory(ref mut p) = load_persister {
        if let SafetensorsPersister::Memory(ref p_save) = save_persister {
            // Get Arc and extract data
            let data_arc = p_save.data().unwrap();
            p.set_data(data_arc.as_ref().clone());
        }
    }
    let result = module2.apply_from(&mut load_persister).unwrap();

    assert!(result.is_success());
    assert!(result.applied.len() > 5);
    assert_eq!(result.errors.len(), 0);

    // Verify data was imported correctly
    let module2_views = module2.collect();
    let encoder_weight = module2_views
        .iter()
        .find(|v| v.full_path() == "encoder.weight")
        .unwrap()
        .to_data();
    assert_eq!(encoder_weight.shape, vec![2, 2, 2]);
}

#[test]
#[cfg(target_has_atomic = "ptr")]
fn test_filtered_export_import() {
    let device = Default::default();
    let module1 = ComplexModule::<TestBackend>::new(&device);
    let mut module2 = ComplexModule::<TestBackend>::new_zeros(&device);

    // Export only encoder tensors
    let mut save_persister = SafetensorsPersisterConfig::new()
        .with_filter(PathFilter::new().with_regex(r"^encoder\..*"))
        .build_memory();
    module1.collect_to(&mut save_persister).unwrap();

    // Import filtered tensors - need to allow partial since we only saved encoder tensors
    let mut load_persister = SafetensorsPersisterConfig::new()
        .allow_partial(true)
        .build_memory();
    if let SafetensorsPersister::Memory(ref mut p) = load_persister {
        if let SafetensorsPersister::Memory(ref p_save) = save_persister {
            // Get Arc and extract data
            let data_arc = p_save.data().unwrap();
            p.set_data(data_arc.as_ref().clone());
        }
    }
    let result = module2.apply_from(&mut load_persister).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 3); // encoder.weight, encoder.bias, encoder.norm
    assert!(result.missing.len() > 0); // decoder and layers tensors are missing
}

#[test]
fn test_metadata_preservation() {
    let device = Default::default();
    let module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Write with metadata
    let mut save_persister = SafetensorsPersisterConfig::new()
        .with_metadata("framework", "burn")
        .with_metadata("version", "0.14.0")
        .with_metadata("model_type", "linear")
        .build_memory();

    module.collect_to(&mut save_persister).unwrap();

    // Verify metadata was saved (would need to add a method to check metadata)
    // For now, just verify the round trip works
    let mut load_persister = SafetensorsPersisterConfig::new().build_memory();
    if let SafetensorsPersister::Memory(ref mut p) = load_persister {
        if let SafetensorsPersister::Memory(ref p_save) = save_persister {
            // Get Arc and extract data
            let data_arc = p_save.data().unwrap();
            p.set_data(data_arc.as_ref().clone());
        }
    }

    let mut module2 = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);
    let result = module2.apply_from(&mut load_persister).unwrap();

    assert!(result.is_success());
}

#[test]
fn test_error_handling() {
    let device = Default::default();

    // Create a module
    let module = LinearConfig::new(2, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Save module
    let mut save_persister = SafetensorsPersisterConfig::new().build_memory();
    module.collect_to(&mut save_persister).unwrap();

    // Try to load into incompatible module (different dimensions)
    let mut incompatible_module = LinearConfig::new(3, 3)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Load without validation - should return errors in the result
    let mut load_persister = SafetensorsPersisterConfig::new()
        .with_validation(false) // Disable validation to get errors in result
        .build_memory();
    if let SafetensorsPersister::Memory(ref mut p) = load_persister {
        if let SafetensorsPersister::Memory(ref p_save) = save_persister {
            // Get Arc and extract data
            let data_arc = p_save.data().unwrap();
            p.set_data(data_arc.as_ref().clone());
        }
    }

    let result = incompatible_module.apply_from(&mut load_persister).unwrap();

    // Should have errors due to shape mismatch
    assert!(!result.errors.is_empty());

    // Try again with validation enabled - should return Err
    let mut load_persister_with_validation = SafetensorsPersisterConfig::new()
        .with_validation(true)
        .build_memory();
    if let SafetensorsPersister::Memory(ref mut p) = load_persister_with_validation {
        if let SafetensorsPersister::Memory(ref p_save) = save_persister {
            // Get Arc and extract data
            let data_arc = p_save.data().unwrap();
            p.set_data(data_arc.as_ref().clone());
        }
    }

    let validation_result = incompatible_module.apply_from(&mut load_persister_with_validation);
    assert!(validation_result.is_err());
}

// Integration tests demonstrating the SafeTensors persister API
#[derive(Module, Debug)]
struct IntegrationTestModel<B: Backend> {
    encoder: IntegrationEncoderModule<B>,
    decoder: IntegrationDecoderModule<B>,
    head: IntegrationHeadModule<B>,
}

#[derive(Module, Debug)]
struct IntegrationEncoderModule<B: Backend> {
    layer1: IntegrationLinearLayer<B>,
    layer2: IntegrationLinearLayer<B>,
    norm: IntegrationNormLayer<B>,
}

#[derive(Module, Debug)]
struct IntegrationDecoderModule<B: Backend> {
    layer1: IntegrationLinearLayer<B>,
    layer2: IntegrationLinearLayer<B>,
    norm: IntegrationNormLayer<B>,
}

#[derive(Module, Debug)]
struct IntegrationHeadModule<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
struct IntegrationLinearLayer<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
}

#[derive(Module, Debug)]
struct IntegrationNormLayer<B: Backend> {
    scale: Param<Tensor<B, 1>>,
    shift: Param<Tensor<B, 1>>,
}

impl<B: Backend> IntegrationTestModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            encoder: IntegrationEncoderModule::new(device),
            decoder: IntegrationDecoderModule::new(device),
            head: IntegrationHeadModule::new(device),
        }
    }
}

impl<B: Backend> IntegrationEncoderModule<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            layer1: IntegrationLinearLayer::new(device, 1),
            layer2: IntegrationLinearLayer::new(device, 2),
            norm: IntegrationNormLayer::new(device),
        }
    }
}

impl<B: Backend> IntegrationDecoderModule<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            layer1: IntegrationLinearLayer::new(device, 3),
            layer2: IntegrationLinearLayer::new(device, 4),
            norm: IntegrationNormLayer::new(device),
        }
    }
}

impl<B: Backend> IntegrationHeadModule<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            weight: Param::from_data([[5.0, 6.0], [7.0, 8.0]], device),
            bias: Param::from_data([9.0, 10.0], device),
        }
    }
}

impl<B: Backend> IntegrationLinearLayer<B> {
    fn new(device: &B::Device, seed: i32) -> Self {
        let weight_data = [
            [seed as f32, (seed + 1) as f32],
            [(seed + 2) as f32, (seed + 3) as f32],
        ];
        let bias_data = [(seed + 4) as f32, (seed + 5) as f32];

        Self {
            weight: Param::from_data(weight_data, device),
            bias: Param::from_data(bias_data, device),
        }
    }
}

impl<B: Backend> IntegrationNormLayer<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            scale: Param::from_data([1.0, 2.0], device),
            shift: Param::from_data([0.1, 0.2], device),
        }
    }
}

#[test]
fn test_integration_basic_usage() {
    let device = Default::default();
    let model = IntegrationTestModel::<TestBackend>::new(&device);

    // Save using new API
    let mut save_persister = SafetensorsPersisterConfig::new()
        .with_metadata("framework", "burn")
        .with_metadata("version", "0.19.0")
        .build_memory();

    // Use collect_to method
    model.collect_to(&mut save_persister).unwrap();

    // Load using new API
    let mut load_persister = SafetensorsPersisterConfig::new().build_memory();
    if let SafetensorsPersister::Memory(ref mut p) = load_persister {
        if let SafetensorsPersister::Memory(ref p_save) = save_persister {
            p.set_data(p_save.data().unwrap().as_ref().clone());
        }
    }

    let mut target_model = IntegrationTestModel::<TestBackend>::new(&device);
    let result = target_model.apply_from(&mut load_persister).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 14); // All tensors should be applied
    assert_eq!(result.errors.len(), 0);
    assert_eq!(result.unused.len(), 0);
}

#[test]
#[cfg(target_has_atomic = "ptr")]
fn test_integration_with_filtering() {
    let device = Default::default();
    let model = IntegrationTestModel::<TestBackend>::new(&device);

    // Save only encoder tensors
    let mut save_persister = SafetensorsPersisterConfig::new()
        .with_filter(PathFilter::new().with_regex(r"^encoder\..*"))
        .with_metadata("subset", "encoder_only")
        .build_memory();

    model.collect_to(&mut save_persister).unwrap();

    // Load into new model - need to allow partial loading since we only saved encoder tensors
    let mut load_persister = SafetensorsPersisterConfig::new()
        .allow_partial(true)
        .build_memory();
    if let SafetensorsPersister::Memory(ref mut p) = load_persister {
        if let SafetensorsPersister::Memory(ref p_save) = save_persister {
            p.set_data(p_save.data().unwrap().as_ref().clone());
        }
    }

    let mut target_model = IntegrationTestModel::<TestBackend>::new(&device);
    let result = target_model.apply_from(&mut load_persister).unwrap();

    // Only encoder tensors should be applied
    assert_eq!(result.applied.len(), 6); // encoder has 6 tensors (2 layers × 2 + norm × 2)

    // Check that only encoder tensors were applied
    for tensor_name in &result.applied {
        assert!(tensor_name.starts_with("encoder."));
    }
}

#[test]
#[cfg(feature = "std")]
fn test_file_based_loading() {
    use std::fs;

    let device = Default::default();
    let module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Create temp file path
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_safetensors.st");

    // Save to file
    let mut save_persister = SafetensorsPersisterConfig::new()
        .with_metadata("test", "file_loading")
        .build_file(&file_path);

    module.collect_to(&mut save_persister).unwrap();

    // Verify file exists
    assert!(file_path.exists());

    // Load from file (will use memory-mapped loading if available)
    let mut load_persister = SafetensorsPersisterConfig::new().build_file(&file_path);

    let mut loaded_module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    let result = loaded_module.apply_from(&mut load_persister).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 2); // weight and bias

    // Clean up
    fs::remove_file(file_path).ok();
}
