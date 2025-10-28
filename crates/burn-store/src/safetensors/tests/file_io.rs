use crate::{ModuleSnapshot, ModuleStore, SafetensorsStore};
use burn_nn::LinearConfig;

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

    module.save_into(&mut save_store).unwrap();

    // Verify file exists
    assert!(file_path.exists());

    // Load from file (will use memory-mapped loading if available)
    let mut load_store = SafetensorsStore::from_file(&file_path);

    let mut loaded_module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    let result = loaded_module.load_from(&mut load_store).unwrap();

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
    let result = module2.load_from(&mut load_store).unwrap();
    assert!(result.is_success());
}

#[test]
#[cfg(feature = "std")]
fn test_forward_pass_preservation_after_save_load() {
    use burn_core as burn;

    use burn_core::module::Module;
    use burn_tensor::Tensor;
    use tempfile::tempdir;

    // Define a test model with forward pass
    #[derive(Module, Debug)]
    struct ForwardTestModel<B: burn_tensor::backend::Backend> {
        linear1: burn_nn::Linear<B>,
        linear2: burn_nn::Linear<B>,
    }

    impl<B: burn_tensor::backend::Backend> ForwardTestModel<B> {
        fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            let x = self.linear1.forward(input);
            let x = burn::tensor::activation::gelu(x);
            self.linear2.forward(x)
        }
    }

    // Define config for the model
    #[derive(burn::config::Config, Debug)]
    struct ForwardTestModelConfig {
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    }

    impl ForwardTestModelConfig {
        fn init<B: burn_tensor::backend::Backend>(
            &self,
            device: &B::Device,
        ) -> ForwardTestModel<B> {
            ForwardTestModel {
                linear1: LinearConfig::new(self.input_size, self.hidden_size)
                    .with_bias(true)
                    .init(device),
                linear2: LinearConfig::new(self.hidden_size, self.output_size)
                    .with_bias(true)
                    .init(device),
            }
        }
    }

    let device = Default::default();

    // Create model config
    let config = ForwardTestModelConfig {
        input_size: 4,
        hidden_size: 8,
        output_size: 2,
    };

    // Initialize model1 with random weights
    let model1 = config.init::<TestBackend>(&device);

    // Create random input
    let input = Tensor::<TestBackend, 2>::random(
        [1, 4],
        burn_tensor::Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    // Forward pass with model1 -> output1
    let output1 = model1.forward(input.clone());

    // Save model1 weights
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("forward_test_model.safetensors");
    let mut save_store = SafetensorsStore::from_file(&path);
    save_store.collect_from(&model1).unwrap();

    // Initialize model2 with different random weights
    let mut model2 = config.init::<TestBackend>(&device);

    // Forward pass with model2 -> output2 (should differ from output1)
    let output2 = model2.forward(input.clone());

    // Verify output2 differs from output1 (different random weights)
    assert!(
        !output1
            .clone()
            .all_close(output2.clone(), Some(1e-6), Some(1e-6)),
        "output2 should differ from output1 (different random initializations)"
    );

    // Load model1 weights into model2
    let mut load_store = SafetensorsStore::from_file(&path);
    let result = load_store.apply_to(&mut model2).unwrap();
    assert!(result.is_success());
    assert_eq!(result.applied.len(), 4); // 2 weights + 2 biases

    // Forward pass with model2 (now has model1 weights) -> output3
    let output3 = model2.forward(input.clone());

    // Verify output3 equals output1 (same weights)
    assert!(
        output1.all_close(output3, Some(1e-6), Some(1e-6)),
        "output3 should equal output1 after loading weights"
    );
}
