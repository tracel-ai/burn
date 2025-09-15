use crate::{ModuleSnapshot, SafetensorsStore};
use burn_core::nn::LinearConfig;

type TestBackend = burn_ndarray::NdArray;

#[test]
fn shape_mismatch_errors() {
    let device = Default::default();

    // Create a module
    let module = LinearConfig::new(2, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Save module
    let mut save_persister = SafetensorsStore::from_bytes(None);
    module.collect_to(&mut save_persister).unwrap();

    // Try to load into incompatible module (different dimensions)
    let mut incompatible_module = LinearConfig::new(3, 3)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Load without validation - should return errors in the result
    let mut load_persister = SafetensorsStore::from_bytes(None).validate(false); // Disable validation to get errors in result
    if let SafetensorsStore::Memory(ref mut p) = load_persister {
        if let SafetensorsStore::Memory(ref p_save) = save_persister {
            // Get Arc and extract data
            let data_arc = p_save.data().unwrap();
            p.set_data(data_arc.as_ref().clone());
        }
    }

    let result = incompatible_module.apply_from(&mut load_persister).unwrap();

    // Should have errors due to shape mismatch
    assert!(!result.errors.is_empty());

    // Try again with validation enabled - should return Err
    let mut load_persister_with_validation = SafetensorsStore::from_bytes(None).validate(true);
    if let SafetensorsStore::Memory(ref mut p) = load_persister_with_validation {
        if let SafetensorsStore::Memory(ref p_save) = save_persister {
            // Get Arc and extract data
            let data_arc = p_save.data().unwrap();
            p.set_data(data_arc.as_ref().clone());
        }
    }

    let validation_result = incompatible_module.apply_from(&mut load_persister_with_validation);
    assert!(validation_result.is_err());
}
