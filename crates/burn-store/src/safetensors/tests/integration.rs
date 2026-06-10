use burn_core as burn;

use crate::{ModuleSnapshot, SafetensorsStore};
use burn_core::module::{Module, Param};
use burn_core::tensor::{Device, Tensor};

// Integration tests demonstrating the SafeTensors store API
#[derive(Module, Debug)]
struct IntegrationTestModel {
    encoder: IntegrationEncoderModule,
    decoder: IntegrationDecoderModule,
    head: IntegrationHeadModule,
}

#[derive(Module, Debug)]
struct IntegrationEncoderModule {
    layer1: IntegrationLinearLayer,
    layer2: IntegrationLinearLayer,
    norm: IntegrationNormLayer,
}

#[derive(Module, Debug)]
struct IntegrationDecoderModule {
    layer1: IntegrationLinearLayer,
    layer2: IntegrationLinearLayer,
    norm: IntegrationNormLayer,
}

#[derive(Module, Debug)]
struct IntegrationHeadModule {
    weight: Param<Tensor<2>>,
    bias: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct IntegrationLinearLayer {
    weight: Param<Tensor<2>>,
    bias: Param<Tensor<1>>,
}

#[derive(Module, Debug)]
struct IntegrationNormLayer {
    scale: Param<Tensor<1>>,
    shift: Param<Tensor<1>>,
}

impl IntegrationTestModel {
    fn new(device: &Device) -> Self {
        Self {
            encoder: IntegrationEncoderModule::new(device),
            decoder: IntegrationDecoderModule::new(device),
            head: IntegrationHeadModule::new(device),
        }
    }
}

impl IntegrationEncoderModule {
    fn new(device: &Device) -> Self {
        Self {
            layer1: IntegrationLinearLayer::new(device, 1),
            layer2: IntegrationLinearLayer::new(device, 2),
            norm: IntegrationNormLayer::new(device),
        }
    }
}

impl IntegrationDecoderModule {
    fn new(device: &Device) -> Self {
        Self {
            layer1: IntegrationLinearLayer::new(device, 3),
            layer2: IntegrationLinearLayer::new(device, 4),
            norm: IntegrationNormLayer::new(device),
        }
    }
}

impl IntegrationHeadModule {
    fn new(device: &Device) -> Self {
        Self {
            weight: Param::from_data([[5.0, 6.0], [7.0, 8.0]], device),
            bias: Param::from_data([9.0, 10.0], device),
        }
    }
}

impl IntegrationLinearLayer {
    fn new(device: &Device, seed: i32) -> Self {
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

impl IntegrationNormLayer {
    fn new(device: &Device) -> Self {
        Self {
            scale: Param::from_data([1.0, 2.0], device),
            shift: Param::from_data([0.1, 0.2], device),
        }
    }
}

#[test]
fn basic_usage() {
    let device = Default::default();
    let model = IntegrationTestModel::new(&device);

    // Save using new API (format, producer and version are automatically added)
    let mut save_store = SafetensorsStore::from_bytes(None).metadata("model_name", "test_model");

    // Use collect_to method
    model.save_into(&mut save_store).unwrap();

    // Load using new API
    let mut load_store = SafetensorsStore::from_bytes(None);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        p.set_data(p_save.data().unwrap().as_ref().clone());
    }

    let mut target_model = IntegrationTestModel::new(&device);
    let result = target_model.load_from(&mut load_store).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 14); // All tensors should be applied
    assert_eq!(result.errors.len(), 0);
    assert_eq!(result.unused.len(), 0);
}

#[test]
#[cfg(target_has_atomic = "ptr")]
fn with_filtering() {
    let device = Default::default();
    let model = IntegrationTestModel::new(&device);

    // Save only encoder tensors using the builder pattern
    let mut save_store = SafetensorsStore::from_bytes(None)
        .with_regex(r"^encoder\..*")
        .metadata("subset", "encoder_only");

    model.save_into(&mut save_store).unwrap();

    // Load into new model - need to allow partial loading since we only saved encoder tensors
    let mut load_store = SafetensorsStore::from_bytes(None).allow_partial(true);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        p.set_data(p_save.data().unwrap().as_ref().clone());
    }

    let mut target_model = IntegrationTestModel::new(&device);
    let result = target_model.load_from(&mut load_store).unwrap();

    // Only encoder tensors should be applied
    assert_eq!(result.applied.len(), 6); // encoder has 6 tensors (2 layers × 2 + norm × 2)

    // Check that only encoder tensors were applied
    for tensor_name in &result.applied {
        assert!(tensor_name.starts_with("encoder."));
    }
}
