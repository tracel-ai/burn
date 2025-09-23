use crate::{ModuleSnapshot, SafetensorsStore};
use burn::nn::LinearConfig;

type TestBackend = burn_ndarray::NdArray;

#[test]
fn default_metadata_included() {
    // Verify that default metadata is automatically included
    let default_metadata = SafetensorsStore::default_metadata();

    // Check that format, producer and version are present
    assert_eq!(default_metadata.get("format").unwrap(), "safetensors");
    assert_eq!(default_metadata.get("producer").unwrap(), "burn");
    assert!(default_metadata.contains_key("version"));

    // The version should be the crate version
    let version = default_metadata.get("version").unwrap();
    assert!(!version.is_empty());
}

#[test]
fn metadata_preservation() {
    let device = Default::default();
    let module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Write with metadata - note that format, producer and version are automatically added
    let mut save_store = SafetensorsStore::from_bytes(None)
        .metadata("model_type", "linear")
        .metadata("custom_field", "test_value");

    module.collect_to(&mut save_store).unwrap();

    // Verify metadata was saved (would need to add a method to check metadata)
    // For now, just verify the round trip works
    let mut load_store = SafetensorsStore::from_bytes(None);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        // Get Arc and extract data
        let data_arc = p_save.data().unwrap();
        p.set_data(data_arc.as_ref().clone());
    }

    let mut module2 = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);
    let result = module2.apply_from(&mut load_store).unwrap();

    assert!(result.is_success());
}

#[test]
fn clear_metadata_removes_all() {
    let device = Default::default();
    let module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Create store with custom metadata, then clear all
    let mut save_store = SafetensorsStore::from_bytes(None)
        .metadata("model_type", "linear")
        .metadata("custom_field", "test_value")
        .clear_metadata(); // Should remove all metadata including defaults

    module.collect_to(&mut save_store).unwrap();

    // Load and verify the module still works (metadata is optional)
    let mut load_store = SafetensorsStore::from_bytes(None);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        let data_arc = p_save.data().unwrap();
        p.set_data(data_arc.as_ref().clone());
    }

    let mut module2 = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);
    let result = module2.apply_from(&mut load_store).unwrap();

    assert!(result.is_success());
}

#[test]
fn clear_then_add_custom_metadata() {
    let device = Default::default();
    let module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Clear all metadata, then add only custom ones
    let mut save_store = SafetensorsStore::from_bytes(None)
        .clear_metadata()
        .metadata("only_custom", "value");

    module.collect_to(&mut save_store).unwrap();

    // Verify round-trip works
    let mut load_store = SafetensorsStore::from_bytes(None);
    if let SafetensorsStore::Memory(ref mut p) = load_store
        && let SafetensorsStore::Memory(ref p_save) = save_store
    {
        let data_arc = p_save.data().unwrap();
        p.set_data(data_arc.as_ref().clone());
    }

    let mut module2 = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);
    let result = module2.apply_from(&mut load_store).unwrap();

    assert!(result.is_success());
}
