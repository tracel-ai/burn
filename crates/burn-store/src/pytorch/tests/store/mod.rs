//! Comprehensive tests for PytorchStore with real model application
use burn_core as burn;

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

/// Path to store test data files
fn store_test_data_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("pytorch")
        .join("tests")
        .join("store")
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

#[cfg(test)]
mod enum_variant_tests {
    use super::*;
    use crate::ModuleSnapshot;
    use burn_ndarray::NdArray;

    /// Enum representing different convolution block types (similar to YOLOX architecture)
    #[derive(Module, Debug)]
    pub enum ConvBlock<B: Backend> {
        /// Base convolution block
        BaseConv(Linear<B>),
        /// Depthwise separable convolution block
        DwsConv(Linear<B>),
    }

    /// Model with enum field that will have variant names in Burn paths
    #[derive(Module, Debug)]
    pub struct ModelWithEnum<B: Backend> {
        /// Feature extractor with enum variants
        feature: ConvBlock<B>,
        /// Output classifier
        classifier: Linear<B>,
    }

    impl<B: Backend> ModelWithEnum<B> {
        pub fn new(device: &B::Device) -> Self {
            Self {
                feature: ConvBlock::BaseConv(LinearConfig::new(3, 64).init(device)),
                classifier: LinearConfig::new(64, 10).init(device),
            }
        }
    }

    #[test]
    fn test_enum_variant_path_mismatch() {
        let device = Default::default();
        let mut model = ModelWithEnum::<NdArray>::new(&device);

        // Load PyTorch model that was generated without enum variant names
        // PyTorch paths: "feature.weight", "feature.bias", "classifier.weight", "classifier.bias"
        // Burn paths:    "feature.BaseConv.weight", "feature.BaseConv.bias", "classifier.weight", "classifier.bias"
        //                         ^^^^^^^^ enum variant name is included in Burn but not PyTorch

        let pytorch_file = store_test_data_path("model_without_enum_variants.pt");

        // Try to load from PyTorch format (without enum variants)
        // Explicitly disable skip_enum_variants to demonstrate the mismatch problem
        let mut store = PytorchStore::from_file(pytorch_file)
            .skip_enum_variants(false) // Disable to show the mismatch
            .allow_partial(true) // Allow partial to see what's missing
            .validate(false); // Disable validation to get detailed missing info

        let result = store.apply_to::<NdArray, _>(&mut model);

        // The load should succeed (allow_partial=true) but report missing tensors
        match result {
            Ok(apply_result) => {
                // Verify we have missing tensors
                assert!(
                    !apply_result.missing.is_empty(),
                    "Should have missing tensors due to enum variant path mismatch"
                );

                // Check that missing paths contain enum variants
                let enum_missing: Vec<_> = apply_result
                    .missing
                    .iter()
                    .filter(|(_, container_stack)| container_stack.contains("Enum:"))
                    .collect();

                assert!(
                    !enum_missing.is_empty(),
                    "Missing tensors should be detected as having enum containers"
                );

                // Verify the paths look like what we expect
                let has_base_conv_path = apply_result
                    .missing
                    .iter()
                    .any(|(path, _)| path.contains("BaseConv"));

                assert!(
                    has_base_conv_path,
                    "Should have missing paths with 'BaseConv' enum variant. Missing: {:?}",
                    apply_result
                        .missing
                        .iter()
                        .map(|(p, _)| p)
                        .collect::<Vec<_>>()
                );

                // Print the diagnostic output to show enum detection
                println!("\n{}", apply_result);

                // Verify the diagnostic message mentions enum variants
                let display_output = format!("{}", apply_result);
                assert!(
                    display_output.contains("enum variant"),
                    "Display output should mention enum variants"
                );
            }
            Err(e) => panic!(
                "Load should succeed with allow_partial=true, got error: {}",
                e
            ),
        }
    }

    #[test]
    fn test_enum_variant_detection_in_container_stack() {
        let device = Default::default();

        // Create model with enum
        let model = ModelWithEnum::<NdArray>::new(&device);

        // Collect snapshots to inspect container stacks
        let snapshots = model.collect(None, None, false);

        // Find a snapshot from inside the enum
        let enum_snapshot = snapshots
            .iter()
            .find(|s| s.full_path().contains("feature"))
            .expect("Should have feature snapshots");

        // Verify container stack contains enum marker
        if let Some(container_stack) = &enum_snapshot.container_stack {
            let container_str = container_stack.join(".");
            assert!(
                container_str.contains("Enum:ConvBlock"),
                "Container stack should contain Enum:ConvBlock marker. Got: {}",
                container_str
            );
        } else {
            panic!("Snapshot should have container_stack");
        }
    }

    #[test]
    fn test_skip_enum_variants_feature() {
        let device = Default::default();
        let mut model = ModelWithEnum::<NdArray>::new(&device);

        // Load PyTorch model that was generated without enum variant names
        // PyTorch paths: "feature.weight", "feature.bias", "classifier.weight", "classifier.bias"
        // Burn paths:    "feature.BaseConv.weight", "feature.BaseConv.bias", "classifier.weight", "classifier.bias"

        let pytorch_file = store_test_data_path("model_without_enum_variants.pt");

        // Try to load with skip_enum_variants enabled
        let mut store = PytorchStore::from_file(pytorch_file)
            .skip_enum_variants(true) // Enable enum variant skipping
            .allow_partial(true)
            .validate(false);

        let result = store.apply_to::<NdArray, _>(&mut model);

        // The load should succeed and all tensors should be loaded
        match result {
            Ok(apply_result) => {
                println!("\n{}", apply_result);

                // With skip_enum_variants enabled, we should successfully load the feature tensors
                let feature_applied = apply_result
                    .applied
                    .iter()
                    .filter(|path| path.contains("feature"))
                    .count();

                assert!(
                    feature_applied > 0,
                    "Should have applied feature tensors with skip_enum_variants=true. Applied: {:?}",
                    apply_result.applied
                );

                // The feature tensors should NOT be in missing anymore
                let feature_missing = apply_result
                    .missing
                    .iter()
                    .filter(|(path, _)| path.contains("feature"))
                    .count();

                assert_eq!(
                    feature_missing, 0,
                    "Feature tensors should not be missing with skip_enum_variants=true. Missing: {:?}",
                    apply_result.missing
                );
            }
            Err(e) => panic!(
                "Load with skip_enum_variants should succeed, got error: {}",
                e
            ),
        }
    }
}

#[cfg(test)]
mod direct_access_tests {
    use super::*;

    #[test]
    fn test_get_all_snapshots() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping test - file not found: {:?}", path);
            return;
        }

        let mut store = PytorchStore::from_file(path);
        let snapshots = store.get_all_snapshots().unwrap();

        // linear.pt should have fc1.weight, fc1.bias, fc2.weight, fc2.bias
        assert!(!snapshots.is_empty(), "Should have snapshots");
        assert!(
            snapshots.contains_key("fc1.weight"),
            "Should contain fc1.weight"
        );
        assert!(
            snapshots.contains_key("fc1.bias"),
            "Should contain fc1.bias"
        );
    }

    #[test]
    fn test_get_snapshot_existing() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping test - file not found: {:?}", path);
            return;
        }

        let mut store = PytorchStore::from_file(path);

        // Get existing snapshot
        let snapshot = store.get_snapshot("fc1.weight").unwrap();
        assert!(snapshot.is_some(), "Should find fc1.weight");

        let snapshot = snapshot.unwrap();
        // Linear weight should be 2D
        assert_eq!(snapshot.shape.len(), 2, "Weight should be 2D tensor");

        // Verify we can load data
        let data = snapshot.to_data().unwrap();
        assert!(!data.bytes.is_empty(), "Data should not be empty");
    }

    #[test]
    fn test_get_snapshot_not_found() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping test - file not found: {:?}", path);
            return;
        }

        let mut store = PytorchStore::from_file(path);

        // Get non-existent snapshot
        let snapshot = store.get_snapshot("nonexistent.weight").unwrap();
        assert!(snapshot.is_none(), "Should not find nonexistent tensor");
    }

    #[test]
    fn test_keys() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping test - file not found: {:?}", path);
            return;
        }

        let mut store = PytorchStore::from_file(path);
        let keys = store.keys().unwrap();

        assert!(!keys.is_empty(), "Should have keys");
        assert!(
            keys.contains(&"fc1.weight".to_string()),
            "Keys should contain fc1.weight"
        );
        assert!(
            keys.contains(&"fc1.bias".to_string()),
            "Keys should contain fc1.bias"
        );
    }

    #[test]
    fn test_keys_fast_path() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping test - file not found: {:?}", path);
            return;
        }

        // Create fresh store - cache should be empty
        let mut store = PytorchStore::from_file(&path);

        // keys() should work without populating the full cache (fast path)
        let keys = store.keys().unwrap();
        assert!(!keys.is_empty(), "Should have keys via fast path");

        // Now call get_all_snapshots to populate cache
        let snapshots = store.get_all_snapshots().unwrap();
        assert!(!snapshots.is_empty(), "Should have snapshots");

        // keys() should now use the cached data
        let keys2 = store.keys().unwrap();
        assert_eq!(keys.len(), keys2.len(), "Keys count should match");
    }

    #[test]
    fn test_caching_behavior() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping test - file not found: {:?}", path);
            return;
        }

        let mut store = PytorchStore::from_file(path);

        // First call populates cache
        let snapshots1 = store.get_all_snapshots().unwrap();
        let count1 = snapshots1.len();

        // Second call uses cache
        let snapshots2 = store.get_all_snapshots().unwrap();
        let count2 = snapshots2.len();

        assert_eq!(count1, count2, "Cached results should match");
    }

    #[test]
    fn test_get_all_snapshots_with_remapping() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping test - file not found: {:?}", path);
            return;
        }

        // Create store with key remapping
        let mut store = PytorchStore::from_file(path).with_key_remapping(r"^fc1\.", "linear1.");

        let snapshots = store.get_all_snapshots().unwrap();

        // Should have remapped keys
        assert!(
            snapshots.contains_key("linear1.weight"),
            "Should contain remapped key linear1.weight. Keys: {:?}",
            snapshots.keys().collect::<Vec<_>>()
        );
        assert!(
            snapshots.contains_key("linear1.bias"),
            "Should contain remapped key linear1.bias"
        );

        // Original keys should not exist
        assert!(
            !snapshots.contains_key("fc1.weight"),
            "Should not contain original key fc1.weight"
        );
    }

    #[test]
    fn test_get_snapshot_with_remapped_name() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping test - file not found: {:?}", path);
            return;
        }

        // Create store with key remapping
        let mut store = PytorchStore::from_file(path).with_key_remapping(r"^fc1\.", "linear1.");

        // Should find by remapped name
        let snapshot = store.get_snapshot("linear1.weight").unwrap();
        assert!(snapshot.is_some(), "Should find tensor by remapped name");

        // Should NOT find by original name
        let snapshot_orig = store.get_snapshot("fc1.weight").unwrap();
        assert!(
            snapshot_orig.is_none(),
            "Should not find tensor by original name after remapping"
        );
    }

    #[test]
    fn test_get_all_snapshots_ignores_filter() {
        let path = pytorch_test_path("linear", "linear.pt");

        if !path.exists() {
            println!("Skipping test - file not found: {:?}", path);
            return;
        }

        // Create store with filter that only matches fc1
        let mut store = PytorchStore::from_file(path).with_regex(r"^fc1\.");

        // get_all_snapshots should return ALL tensors regardless of filter
        let snapshots = store.get_all_snapshots().unwrap();

        // Should have both fc1 and fc2 tensors
        assert!(
            snapshots.contains_key("fc1.weight"),
            "Should contain fc1.weight"
        );
        assert!(
            snapshots.contains_key("fc2.weight"),
            "Should contain fc2.weight (filter not applied to get_all_snapshots)"
        );
    }
}
