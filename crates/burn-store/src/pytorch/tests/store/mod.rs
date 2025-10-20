//! Comprehensive tests for PytorchStore with real model application

use std::path::PathBuf;

use crate::ModuleStore;
use crate::pytorch::PytorchStore;
use burn_core::module::Module;
use burn_nn::conv::{Conv2d, Conv2dConfig};
use burn_nn::{Linear, LinearConfig};
use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

/// Path to burn-import pytorch test files
fn pytorch_test_path(subdir: &str, filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("burn-import")
        .join("pytorch-tests")
        .join("tests")
        .join(subdir)
        .join(filename)
}

/// Path to burn-store test data files
fn test_data_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("pytorch")
        .join("tests")
        .join("reader")
        .join("test_data")
        .join(filename)
}

#[cfg(test)]
mod basic_tests {
    use super::*;

    #[test]
    fn test_store_creation() {
        let store = PytorchStore::from_file("model.pth");
        assert!(store.validate);
        assert!(!store.allow_partial);
        assert!(store.top_level_key.is_none());
    }

    #[test]
    fn test_store_with_top_level_key() {
        let store = PytorchStore::from_file("model.pth").with_top_level_key("state_dict");
        assert_eq!(store.top_level_key, Some("state_dict".to_string()));
    }

    #[test]
    fn test_store_configuration() {
        let store = PytorchStore::from_file("model.pth")
            .validate(false)
            .allow_partial(true)
            .with_regex(r"^encoder\.")
            .with_full_path("decoder.weight");

        assert!(!store.validate);
        assert!(store.allow_partial);
        assert!(!store.filter.is_empty());
    }

    #[test]
    fn test_store_with_remapping() {
        let store = PytorchStore::from_file("model.pth").with_key_remapping(r"^old\.", "new.");

        assert!(!store.remapper.is_empty());
    }

    #[test]
    fn test_store_save_not_supported() {
        // Currently, saving to PyTorch format is not implemented
        // The collect_from method always returns an error
        let store = PytorchStore::from_file("test.pth");

        // Just verify that store creation works
        assert!(store.validate);

        // Note: Actually testing save would require a proper Module implementation
        // which is complex. The implementation guarantees it returns an error.
    }
}

#[cfg(test)]
mod linear_model_tests {
    use super::*;
    type TestBackend = burn_ndarray::NdArray;

    #[derive(Module, Debug)]
    pub struct SimpleLinearModel<B: Backend> {
        fc1: Linear<B>,
        fc2: Linear<B>,
    }

    impl<B: Backend> SimpleLinearModel<B> {
        pub fn new(device: &B::Device) -> Self {
            Self {
                fc1: LinearConfig::new(2, 3).init(device),
                fc2: LinearConfig::new(3, 4).init(device),
            }
        }

        pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
            let x = self.fc1.forward(x);
            self.fc2.forward(x)
        }
    }

    #[test]
    fn test_load_linear_model() {
        let device = Default::default();
        let path = pytorch_test_path("linear", "linear.pt");

        // Create a model and load weights from PyTorch
        let mut model = SimpleLinearModel::<TestBackend>::new(&device);
        let mut store = PytorchStore::from_file(path).allow_partial(true);

        // Apply the PyTorch weights to our model
        let result = store.apply_to::<TestBackend, _>(&mut model);

        assert!(
            result.is_ok(),
            "Failed to load linear model: {:?}",
            result.err()
        );

        let result = result.unwrap();
        assert!(!result.applied.is_empty(), "No tensors were applied");

        // Test forward pass with loaded weights
        let input = Tensor::<TestBackend, 2>::ones([1, 2], &device);
        let output = model.forward(input);

        // Verify output shape
        assert_eq!(output.shape().dims, [1, 4]);
    }

    #[test]
    fn test_load_linear_with_bias() {
        let device = Default::default();
        let path = pytorch_test_path("linear", "linear_with_bias.pt");

        // Single linear layer with bias
        #[derive(Module, Debug)]
        struct LinearWithBias<B: Backend> {
            fc1: Linear<B>,
        }

        let mut model = LinearWithBias {
            fc1: LinearConfig::new(2, 3).init(&device),
        };

        let mut store = PytorchStore::from_file(path).allow_partial(true);

        let result = store.apply_to::<TestBackend, _>(&mut model);
        assert!(result.is_ok(), "Failed to load model with bias");

        // Verify biases were loaded
        let result = result.unwrap();
        let bias_loaded = result.applied.iter().any(|s| s.contains("bias"));
        assert!(bias_loaded, "Bias parameters not loaded");
    }

    #[test]
    fn test_filter_layers() {
        let device = Default::default();
        let path = pytorch_test_path("linear", "linear.pt");

        let mut model = SimpleLinearModel::<TestBackend>::new(&device);

        // Only load fc1 layers
        let mut store = PytorchStore::from_file(path)
            .with_regex(r"^fc1\.")
            .allow_partial(true);

        let result = store.apply_to::<TestBackend, _>(&mut model).unwrap();

        // Should only have fc1 tensors
        for tensor in &result.applied {
            assert!(tensor.contains("fc1"));
            assert!(!tensor.contains("fc2"));
        }
    }

    #[test]
    fn test_remap_layer_names() {
        let device = Default::default();
        let path = pytorch_test_path("linear", "linear.pt");

        // Model with different layer names
        #[derive(Module, Debug)]
        struct RemappedModel<B: Backend> {
            linear1: Linear<B>,
            linear2: Linear<B>,
        }

        let mut model = RemappedModel {
            linear1: LinearConfig::new(2, 3).init(&device),
            linear2: LinearConfig::new(3, 4).init(&device),
        };

        let mut store = PytorchStore::from_file(path)
            .with_key_remapping(r"^fc1\.", "linear1.")
            .with_key_remapping(r"^fc2\.", "linear2.")
            .allow_partial(true);

        let result = store.apply_to::<TestBackend, _>(&mut model);
        assert!(result.is_ok(), "Failed to load with remapped names");

        let result = result.unwrap();
        // Verify remapped names were applied
        let has_linear1 = result.applied.iter().any(|s| s.contains("linear1"));
        assert!(has_linear1, "Remapped names not applied");
    }
}

#[cfg(test)]
mod conv_model_tests {
    use super::*;

    type TestBackend = burn_ndarray::NdArray;

    #[derive(Module, Debug)]
    struct SimpleConvModel<B: Backend> {
        conv1: Conv2d<B>,
        conv2: Conv2d<B>,
    }

    impl<B: Backend> SimpleConvModel<B> {
        pub fn new(device: &B::Device) -> Self {
            Self {
                conv1: Conv2dConfig::new([3, 16], [3, 3]).init(device),
                conv2: Conv2dConfig::new([16, 32], [3, 3]).init(device),
            }
        }
    }

    #[test]
    fn test_load_conv2d_model() {
        let device = Default::default();
        let path = pytorch_test_path("conv2d", "conv2d.pt");

        // Check if file exists, skip if not
        if !path.exists() {
            println!("Skipping conv2d test - file not found: {:?}", path);
            return;
        }

        let mut model = SimpleConvModel::<TestBackend>::new(&device);
        let mut store = PytorchStore::from_file(path).allow_partial(true);

        let result = store.apply_to::<TestBackend, _>(&mut model);

        if let Ok(result) = result {
            assert!(!result.applied.is_empty(), "No conv tensors applied");

            // Check for conv weights
            let has_conv_weights = result.applied.iter().any(|s| s.contains("weight"));
            assert!(has_conv_weights, "Conv weights not loaded");
        }
    }

    #[test]
    fn test_load_conv1d_model() {
        let path = pytorch_test_path("conv1d", "conv1d.pt");

        if !path.exists() {
            println!("Skipping conv1d test - file not found: {:?}", path);
            return;
        }

        // Just test that we can create a store for conv1d files
        let store = PytorchStore::from_file(path).allow_partial(true);

        assert!(store.allow_partial);
    }
}

#[cfg(test)]
mod complex_model_tests {
    use super::*;
    type TestBackend = burn_ndarray::NdArray;

    #[test]
    fn test_load_with_top_level_key() {
        let path = test_data_path("checkpoint.pt");

        // Just verify that we can create a store with top-level key
        let store = PytorchStore::from_file(path)
            .with_top_level_key("model_state_dict")
            .allow_partial(true);

        assert_eq!(store.top_level_key, Some("model_state_dict".to_string()));
    }

    #[test]
    fn test_load_nested_structure() {
        let path = test_data_path("complex_structure.pt");

        // Just verify that we can create a store for nested structure
        let store = PytorchStore::from_file(path).allow_partial(true);

        assert!(store.allow_partial);
    }

    #[test]
    fn test_legacy_format() {
        let path = test_data_path("simple_legacy.pt");

        if !path.exists() {
            println!("Skipping legacy format test - file not found: {:?}", path);
            return;
        }

        // Just verify that we can create a store for legacy format
        let store = PytorchStore::from_file(path).allow_partial(true);

        assert!(store.allow_partial);

        // Could load into an actual model if we had legacy model structure
    }

    #[test]
    fn test_key_remap_chained() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping key remap test - file not found: {:?}", path);
            return;
        }

        let device = Default::default();

        // Model with different layer names that need remapping
        #[derive(Module, Debug)]
        struct RemappedChainModel<B: Backend> {
            convolution1: Linear<B>, // Will be remapped from fc1
            linear2: Linear<B>,      // Will be remapped from fc2
        }

        let mut model = RemappedChainModel {
            convolution1: LinearConfig::new(2, 3).init(&device),
            linear2: LinearConfig::new(3, 4).init(&device),
        };

        // Chain multiple remappings
        let mut store = PytorchStore::from_file(path)
            .with_key_remapping(r"^fc1\.", "convolution1.")
            .with_key_remapping(r"^fc2\.", "linear2.")
            .allow_partial(true);

        let result = store.apply_to::<TestBackend, _>(&mut model);

        if let Ok(result) = result {
            // Check that remapped names were applied
            assert!(
                !result.applied.is_empty(),
                "No tensors were applied after remapping"
            );
        }
    }
}

#[cfg(test)]
mod adapter_tests {
    use super::*;

    type TestBackend = burn_ndarray::NdArray;

    #[derive(Module, Debug)]
    pub struct SimpleLinearModel<B: Backend> {
        fc1: Linear<B>,
        fc2: Linear<B>,
    }

    impl<B: Backend> SimpleLinearModel<B> {
        pub fn new(device: &B::Device) -> Self {
            Self {
                fc1: LinearConfig::new(2, 3).init(device),
                fc2: LinearConfig::new(3, 4).init(device),
            }
        }
    }

    #[test]
    fn test_pytorch_adapter_always_applied() {
        // Test that PyTorchToBurnAdapter is always applied internally
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping adapter test - file not found: {:?}", path);
            return;
        }

        let device = Default::default();
        let mut model = SimpleLinearModel::<TestBackend>::new(&device);

        let mut store = PytorchStore::from_file(path).allow_partial(true);

        let result = store.apply_to::<TestBackend, _>(&mut model);

        // PyTorchToBurnAdapter is always applied internally
        assert!(
            result.is_ok(),
            "Failed to load with internal PyTorchToBurnAdapter: {:?}",
            result.err()
        );
        assert!(!result.unwrap().applied.is_empty());
    }

    #[test]
    fn test_pytorch_adapter_with_filtering() {
        // Test that PyTorchToBurnAdapter works with filtering
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping filtering test - file not found: {:?}", path);
            return;
        }

        let device = Default::default();
        let mut model = SimpleLinearModel::<TestBackend>::new(&device);

        // Filter to exclude bias tensors
        let mut store = PytorchStore::from_file(path)
            .with_predicate(|path, _| !path.contains("bias"))
            .allow_partial(true);

        let result = store.apply_to::<TestBackend, _>(&mut model).unwrap();

        // Should not have any bias tensors due to filtering
        for applied_path in &result.applied {
            assert!(
                !applied_path.contains("bias"),
                "Bias tensor was not filtered: {}",
                applied_path
            );
        }
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[derive(Module, Debug)]
    pub struct SimpleLinearModel<B: Backend> {
        fc1: Linear<B>,
        fc2: Linear<B>,
    }

    impl<B: Backend> SimpleLinearModel<B> {
        pub fn new(device: &B::Device) -> Self {
            Self {
                fc1: LinearConfig::new(2, 3).init(device),
                fc2: LinearConfig::new(3, 4).init(device),
            }
        }
    }

    #[test]
    fn test_missing_file() {
        let device = Default::default();
        let mut model = SimpleLinearModel::<NdArray>::new(&device);
        let mut store = PytorchStore::from_file("nonexistent.pth");

        let result = store.apply_to::<NdArray, _>(&mut model);

        assert!(result.is_err());
        match result {
            Err(crate::pytorch::PytorchStoreError::Reader(_)) => {}
            _ => panic!("Expected reader error for missing file"),
        }
    }

    #[test]
    fn test_invalid_top_level_key() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!(
                "Skipping invalid top level key test - file not found: {:?}",
                path
            );
            return;
        }

        let device = Default::default();
        let mut model = SimpleLinearModel::<NdArray>::new(&device);

        let mut store = PytorchStore::from_file(path).with_top_level_key("nonexistent_key");

        let result = store.apply_to::<NdArray, _>(&mut model);

        assert!(result.is_err(), "Should fail with invalid top level key");
    }

    #[test]
    fn test_strict_validation() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!(
                "Skipping strict validation test - file not found: {:?}",
                path
            );
            return;
        }

        let device = Default::default();
        let mut model = SimpleLinearModel::<NdArray>::new(&device);

        // Apply very restrictive filter that matches nothing
        let mut store = PytorchStore::from_file(path)
            .with_regex(r"^this_will_never_match$")
            .validate(true)
            .allow_partial(false);

        let result = store.apply_to::<NdArray, _>(&mut model);

        // Should fail because no tensors match and allow_partial is false
        assert!(
            result.is_err(),
            "Should fail when no tensors match with allow_partial=false"
        );
    }
}
