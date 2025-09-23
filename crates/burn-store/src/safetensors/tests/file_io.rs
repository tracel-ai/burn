use crate::{ModuleSnapshot, SafetensorsStore};
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
