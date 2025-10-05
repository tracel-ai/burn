use crate::{ModuleSnapshot, ModuleSnapshoter, SafetensorsStore};
use burn::nn::LinearConfig;

type TestBackend = burn_ndarray::NdArray;

#[test]
#[cfg(feature = "std")]
fn file_based_loading() {
    use std::fs;

    let device = Default::default();
    let module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Create temp file path
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("test_safetensors.st");

    // Save to file
    let mut save_store = SafetensorsStore::from_file(&file_path).metadata("test", "file_loading");

    module.collect_to(&mut save_store).unwrap();

    // Verify file exists
    assert!(file_path.exists());

    // Load from file (will use memory-mapped loading if available)
    let mut load_store = SafetensorsStore::from_file(&file_path);

    let mut loaded_module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    let result = loaded_module.apply_from(&mut load_store).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 2); // weight and bias

    // Clean up
    fs::remove_file(file_path).ok();
}

#[test]
#[cfg(feature = "std")]
fn test_store_overwrite_protection() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Create temp directory and file path (file doesn't exist yet)
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_model.safetensors");

    // First save - should succeed
    let mut save_store = SafetensorsStore::from_file(&path);
    save_store.collect_from(&module).unwrap();
    assert!(path.exists());

    // Second save without overwrite flag - should fail
    let mut save_store2 = SafetensorsStore::from_file(&path);
    let result = save_store2.collect_from(&module);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("File already exists")
    );

    // Third save with overwrite flag - should succeed
    let mut save_store3 = SafetensorsStore::from_file(&path).overwrite(true);
    save_store3.collect_from(&module).unwrap();

    // Verify file still exists and is valid
    let mut load_store = SafetensorsStore::from_file(&path);
    let mut module2 = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);
    let result = load_store.apply_to(&mut module2).unwrap();
    assert!(result.is_success());
}

#[test]
#[cfg(feature = "std")]
fn test_store_overwrite_with_metadata() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Create temp directory and file path
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_model_metadata.safetensors");

    // First save with v1 metadata and overwrite enabled
    let mut save_store = SafetensorsStore::from_file(&path)
        .metadata("model_version", "v1")
        .overwrite(true);
    save_store.collect_from(&module).unwrap();

    // Second save with v2 metadata and overwrite enabled
    let mut save_store2 = SafetensorsStore::from_file(&path)
        .metadata("model_version", "v2")
        .overwrite(true);
    save_store2.collect_from(&module).unwrap();

    // Load and verify the metadata was updated to v2
    let mut load_store = SafetensorsStore::from_file(&path);
    // Since we can't easily access metadata after loading, we just verify the file loads successfully
    let mut module2 = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);
    let result = module2.apply_from(&mut load_store).unwrap();
    assert!(result.is_success());
}
