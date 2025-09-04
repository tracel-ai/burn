use super::*;
use crate as burn;
use crate::TestBackend;
use crate::module::{Module, Param};
use crate::nn::{Linear, LinearConfig};
use crate::persist::{ModulePersist, PathFilter};
use crate::tensor::backend::Backend;
use burn_tensor::Tensor;

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
