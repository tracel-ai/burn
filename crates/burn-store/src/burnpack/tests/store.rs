#[cfg(feature = "std")]
use crate::KeyRemapper;
use crate::burnpack::store::BurnpackStore;
use crate::{ModuleSnapshot, ModuleStore, PathFilter};

use burn_core as burn;
use burn_core::module::{Module, Param};
use burn_tensor::{Tensor, backend::Backend};

type TestBackend = burn_ndarray::NdArray;

#[derive(Module, Debug)]
struct TestModule<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Param<Tensor<B, 1>>,
    nested: NestedModule<B>,
}

#[derive(Module, Debug)]
struct NestedModule<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> TestModule<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            weight: Param::from_data([[1.0, 2.0], [3.0, 4.0]], device),
            bias: Param::from_data([0.1, 0.2], device),
            nested: NestedModule {
                gamma: Param::from_data([1.0, 1.0], device),
                beta: Param::from_data([0.0, 0.0], device),
            },
        }
    }

    fn new_zeros(device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([2, 2], device)),
            bias: Param::from_tensor(Tensor::zeros([2], device)),
            nested: NestedModule {
                gamma: Param::from_tensor(Tensor::zeros([2], device)),
                beta: Param::from_tensor(Tensor::zeros([2], device)),
            },
        }
    }

    fn new_uninitialized(device: &B::Device) -> Self {
        use burn_core::module::ParamId;
        let device_clone = device.clone();
        let device_clone2 = device.clone();
        let device_clone3 = device.clone();
        let device_clone4 = device.clone();

        Self {
            weight: Param::uninitialized(
                ParamId::new(),
                move |d, _| Tensor::zeros([2, 2], d),
                device_clone,
                true,
                [2, 2].into(),
            ),
            bias: Param::uninitialized(
                ParamId::new(),
                move |d, _| Tensor::zeros([2], d),
                device_clone2,
                true,
                [2].into(),
            ),
            nested: NestedModule {
                gamma: Param::uninitialized(
                    ParamId::new(),
                    move |d, _| Tensor::zeros([2], d),
                    device_clone3,
                    true,
                    [2].into(),
                ),
                beta: Param::uninitialized(
                    ParamId::new(),
                    move |d, _| Tensor::zeros([2], d),
                    device_clone4,
                    true,
                    [2].into(),
                ),
            },
        }
    }
}

#[test]
fn test_store_from_bytes_round_trip() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save to bytes
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load from bytes
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    // Verify success
    assert!(result.is_success());
    assert_eq!(result.applied.len(), 4); // weight, bias, nested.gamma, nested.beta
    assert!(result.errors.is_empty());

    // Verify data was loaded correctly
    let weight1 = module.weight.val().to_data().to_vec::<f32>().unwrap();
    let weight2 = module2.weight.val().to_data().to_vec::<f32>().unwrap();
    assert_eq!(weight1, weight2);
}

#[test]
fn test_store_with_metadata() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save with metadata
    let mut save_store = BurnpackStore::from_bytes(None)
        .metadata("version", "1.0.0")
        .metadata("model_name", "test_model")
        .metadata("author", "burn_team");

    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load and verify metadata is preserved
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 4);
}

#[test]
#[cfg(feature = "std")]
fn test_store_with_path_filter() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save all tensors
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load with filter - only load weight and bias (not nested)
    let mut load_store = BurnpackStore::from_bytes(Some(bytes)).with_regex("^(weight|bias)$");

    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 2); // Only weight and bias
    assert_eq!(result.skipped.len(), 2); // nested.gamma and nested.beta skipped

    // Verify weight and bias were loaded
    let weight2 = module2.weight.val().to_data().to_vec::<f32>().unwrap();
    assert_eq!(weight2, vec![1.0, 2.0, 3.0, 4.0]);

    // Verify nested module was NOT loaded (should still be zeros)
    let gamma2 = module2
        .nested
        .gamma
        .val()
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(gamma2, vec![0.0, 0.0]);
}

#[test]
#[cfg(feature = "std")]
fn test_store_with_key_remapping() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save with original names
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load with remapping: nested.gamma -> nested.new_gamma, nested.beta -> nested.new_beta
    let remapper = KeyRemapper::new()
        .add_pattern(r"nested\.gamma", "nested.new_gamma")
        .unwrap()
        .add_pattern(r"nested\.beta", "nested.new_beta")
        .unwrap();

    let mut load_store = BurnpackStore::from_bytes(Some(bytes))
        .remap(remapper)
        .allow_partial(true);

    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    // The remapping should cause missing tensors since names don't match
    assert_eq!(result.applied.len(), 2); // Only weight and bias match
    assert_eq!(result.unused.len(), 2); // nested.new_gamma and nested.new_beta are unused
    assert_eq!(result.missing.len(), 2); // nested.gamma and nested.beta are missing
}

#[test]
fn test_store_allow_partial() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save only weight and bias
    let filter = PathFilter::new()
        .with_full_path("weight")
        .with_full_path("bias");
    let mut save_store = BurnpackStore::from_bytes(None).with_filter(filter);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load with allow_partial
    let mut load_store = BurnpackStore::from_bytes(Some(bytes)).allow_partial(true);

    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 2);
    assert_eq!(result.missing.len(), 2); // nested.gamma and nested.beta are missing but that's OK

    // Verify loaded tensors
    let weight2 = module2.weight.val().to_data().to_vec::<f32>().unwrap();
    assert_eq!(weight2, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_store_match_all() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save with match_all filter (should save everything)
    let mut save_store = BurnpackStore::from_bytes(None).match_all();
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load everything
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 4);
    assert!(result.errors.is_empty());
    assert!(result.missing.is_empty());
    assert!(result.unused.is_empty());
}

#[test]
fn test_store_with_full_path() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save everything
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load only specific tensors by full path
    let mut load_store = BurnpackStore::from_bytes(Some(bytes))
        .with_full_path("weight")
        .with_full_path("nested.gamma");

    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 2); // Only weight and nested.gamma
    assert_eq!(result.skipped.len(), 2); // bias and nested.beta skipped
}

#[test]
#[cfg(feature = "std")]
fn test_store_chain_multiple_patterns() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save with chained metadata and filters
    let mut save_store = BurnpackStore::from_bytes(None)
        .metadata("version", "1.0")
        .metadata("format", "burnpack")
        .with_regex(r"^(weight|nested\.)")
        .match_all(); // This overrides the previous filter

    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load everything since match_all was called last
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 4); // All tensors loaded
}

#[test]
#[cfg(feature = "std")]
fn test_store_with_remap_pattern() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save normally
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load with single remap pattern using the convenience method
    let mut load_store = BurnpackStore::from_bytes(Some(bytes))
        .with_remap_pattern(r"^nested\.", "sub_module.")
        .allow_partial(true);

    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    // After remapping, nested.* becomes sub_module.*, which won't match
    assert_eq!(result.applied.len(), 2); // Only weight and bias
    assert_eq!(result.unused.len(), 2); // sub_module.gamma and sub_module.beta unused
}

#[test]
fn test_store_default_metadata() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save without adding custom metadata
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Verify default metadata is included
    // We can't directly inspect metadata from bytes, but we can verify
    // that the model loads successfully which means metadata was written correctly
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
}

#[test]
fn test_store_default_metadata_with_custom() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save with custom metadata (should preserve defaults)
    let mut save_store = BurnpackStore::from_bytes(None)
        .metadata("custom_field", "custom_value")
        .metadata("author", "test_author");
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load and verify it works (metadata including defaults was saved)
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
}

#[test]
fn test_store_clear_metadata() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save with cleared metadata (no defaults)
    let mut save_store = BurnpackStore::from_bytes(None).clear_metadata();
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Verify it still loads correctly
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
}

#[test]
fn test_store_validate_enabled() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save normally
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load with validation enabled (default)
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
    assert!(result.errors.is_empty());
}

#[test]
fn test_store_validate_disabled() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save normally
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load with validation disabled
    let mut load_store = BurnpackStore::from_bytes(Some(bytes)).validate(false);
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    // Should still succeed
    assert!(result.is_success());
}

#[test]
fn test_store_allow_partial_missing_tensors() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save only weight (not bias or nested)
    let filter = PathFilter::new().with_full_path("weight");
    let mut save_store = BurnpackStore::from_bytes(None).with_filter(filter);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Try to load without allow_partial - should fail due to missing tensors
    let mut load_store = BurnpackStore::from_bytes(Some(bytes.clone()));
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2);

    // Should fail because of missing tensors
    assert!(result.is_err());

    // Now try with allow_partial - should succeed
    let mut load_store = BurnpackStore::from_bytes(Some(bytes)).allow_partial(true);
    let mut module3 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module3).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 1); // Only weight
    assert!(!result.missing.is_empty()); // Has missing tensors
}

#[test]
#[cfg(feature = "std")]
fn test_store_file_round_trip() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Create temp directory and file path
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_file_round_trip.bpk");

    // Save to file
    let mut save_store = BurnpackStore::from_file(&path).metadata("test", "value");
    save_store.collect_from(&module).unwrap();

    // Verify file exists
    assert!(path.exists());

    // Load from file
    let mut load_store = BurnpackStore::from_file(&path);
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 4);

    // Verify data
    let weight1 = module.weight.val().to_data().to_vec::<f32>().unwrap();
    let weight2 = module2.weight.val().to_data().to_vec::<f32>().unwrap();
    assert_eq!(weight1, weight2);
}

#[test]
#[cfg(feature = "std")]
fn test_store_overwrite_protection() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Create temp directory and file path (file doesn't exist yet)
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_model.bpk");

    // First save - should succeed
    let mut save_store = BurnpackStore::from_file(&path);
    save_store.collect_from(&module).unwrap();
    assert!(path.exists());

    // Second save without overwrite flag - should fail
    let mut save_store2 = BurnpackStore::from_file(&path);
    let result = save_store2.collect_from(&module);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("File already exists")
    );

    // Third save with overwrite flag - should succeed
    let mut save_store3 = BurnpackStore::from_file(&path).overwrite(true);
    save_store3.collect_from(&module).unwrap();

    // Verify file still exists and is valid
    let mut load_store = BurnpackStore::from_file(&path);
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();
    assert!(result.is_success());
}

#[test]
#[cfg(feature = "std")]
fn test_store_overwrite_with_metadata() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Create temp directory and file path
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_model_metadata.bpk");

    // First save with v1 metadata
    let mut save_store = BurnpackStore::from_file(&path)
        .metadata("version", "1.0")
        .overwrite(true);
    save_store.collect_from(&module).unwrap();

    // Second save with v2 metadata and overwrite enabled
    let mut save_store2 = BurnpackStore::from_file(&path)
        .metadata("version", "2.0")
        .overwrite(true);
    save_store2.collect_from(&module).unwrap();

    // Verify file loads correctly
    let mut load_store = BurnpackStore::from_file(&path);
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();
    assert!(result.is_success());
}

#[test]
#[cfg(feature = "std")]
fn test_store_auto_extension_default() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Create temp directory
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("model");

    // Save without extension - should auto-append .bpk
    let mut save_store = BurnpackStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    // Verify that model.bpk was created
    let expected_path = temp_dir.path().join("model.bpk");
    assert!(expected_path.exists());
    assert!(!path.exists()); // Original path without extension should not exist

    // Load using the path without extension - should work
    let mut load_store = BurnpackStore::from_file(&path);
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();
    assert!(result.is_success());
}

#[test]
#[cfg(feature = "std")]
fn test_store_auto_extension_with_existing_extension() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Create temp directory
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("model.bpk");

    // Save with .bpk extension - should not double append
    let mut save_store = BurnpackStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    // Verify that only model.bpk was created
    assert!(path.exists());
    let double_ext_path = temp_dir.path().join("model.bpk.bpk");
    assert!(!double_ext_path.exists());

    // Load and verify
    let mut load_store = BurnpackStore::from_file(&path);
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();
    assert!(result.is_success());
}

#[test]
#[cfg(feature = "std")]
fn test_store_auto_extension_with_custom_extension() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Create temp directory
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("model.mpk");

    // Save with .mpk extension - should preserve it
    let mut save_store = BurnpackStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    // Verify that model.mpk was created (not model.mpk.bpk)
    assert!(path.exists());
    let burnpack_path = temp_dir.path().join("model.mpk.bpk");
    assert!(!burnpack_path.exists());

    // Load and verify
    let mut load_store = BurnpackStore::from_file(&path);
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();
    assert!(result.is_success());
}

#[test]
#[cfg(feature = "std")]
fn test_store_auto_extension_disabled() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Create temp directory
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("model");

    // Save with auto_extension disabled - should use exact path
    let mut save_store = BurnpackStore::from_file(&path).auto_extension(false);
    save_store.collect_from(&module).unwrap();

    // Verify that "model" (without extension) was created
    assert!(path.exists());
    let burnpack_path = temp_dir.path().join("model.bpk");
    assert!(!burnpack_path.exists());

    // Load with auto_extension disabled
    let mut load_store = BurnpackStore::from_file(&path).auto_extension(false);
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();
    assert!(result.is_success());
}

#[test]
#[cfg(feature = "std")]
fn test_partial_loading_preserves_lazy_initialization() {
    use tempfile::tempdir;

    let device = Default::default();

    // Create and save a full module
    let module = TestModule::<TestBackend>::new(&device);
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("model.bpk");

    let mut save_store = BurnpackStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    // Create an uninitialized module (all params lazy)
    let mut load_module = TestModule::<TestBackend>::new_uninitialized(&device);

    // Before loading: verify ALL params are uninitialized (lazy)
    assert!(
        !load_module.weight.is_initialized(),
        "weight should be uninitialized before loading"
    );
    assert!(
        !load_module.bias.is_initialized(),
        "bias should be uninitialized before loading"
    );
    assert!(
        !load_module.nested.gamma.is_initialized(),
        "nested.gamma should be uninitialized before loading"
    );
    assert!(
        !load_module.nested.beta.is_initialized(),
        "nested.beta should be uninitialized before loading"
    );

    // Partial load: only load weight and bias (skip nested.*)
    let filter = PathFilter::new().with_regex("^(weight|bias)$");
    let mut load_store = BurnpackStore::from_file(&path).filter(filter);
    let result = load_module.load_from(&mut load_store).unwrap();

    // Verify only weight and bias were loaded
    assert_eq!(result.applied.len(), 2);
    assert!(result.applied.contains(&"weight".to_string()));
    assert!(result.applied.contains(&"bias".to_string()));
    assert_eq!(result.skipped.len(), 2);
    assert!(result.skipped.contains(&"nested.gamma".to_string()));
    assert!(result.skipped.contains(&"nested.beta".to_string()));

    // After loading: verify loaded params are initialized, skipped remain lazy
    assert!(
        load_module.weight.is_initialized(),
        "weight should be initialized after loading"
    );
    assert!(
        load_module.bias.is_initialized(),
        "bias should be initialized after loading"
    );
    assert!(
        !load_module.nested.gamma.is_initialized(),
        "nested.gamma should remain uninitialized (was skipped)"
    );
    assert!(
        !load_module.nested.beta.is_initialized(),
        "nested.beta should remain uninitialized (was skipped)"
    );

    // Verify the loaded values are correct
    let weight_data = load_module.weight.val().to_data().to_vec::<f32>().unwrap();
    assert_eq!(weight_data, vec![1.0, 2.0, 3.0, 4.0]);

    let bias_data = load_module.bias.val().to_data().to_vec::<f32>().unwrap();
    assert_eq!(bias_data, vec![0.1, 0.2]);

    // Now check that nested params can still be initialized on first access
    let gamma_data = load_module
        .nested
        .gamma
        .val()
        .to_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(gamma_data, vec![0.0, 0.0]); // Initialized to zeros via the init function

    // After accessing, they should be initialized
    assert!(
        load_module.nested.gamma.is_initialized(),
        "nested.gamma should be initialized after first access"
    );
}

// Model with forward pass for testing weight preservation
#[derive(Module, Debug)]
struct ForwardTestModel<B: Backend> {
    linear1: burn_nn::Linear<B>,
    linear2: burn_nn::Linear<B>,
}

impl<B: Backend> ForwardTestModel<B> {
    /// Forward pass: input -> linear1 -> gelu -> linear2
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(input);
        let x = burn::tensor::activation::gelu(x);
        self.linear2.forward(x)
    }
}

#[derive(burn::config::Config, Debug)]
struct ForwardTestModelConfig {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
}

impl ForwardTestModelConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ForwardTestModel<B> {
        ForwardTestModel {
            linear1: burn_nn::LinearConfig::new(self.input_size, self.hidden_size)
                .with_bias(true)
                .init(device),
            linear2: burn_nn::LinearConfig::new(self.hidden_size, self.output_size)
                .with_bias(true)
                .init(device),
        }
    }
}

#[test]
#[cfg(feature = "std")]
fn test_forward_pass_preservation_after_save_load() {
    use tempfile::tempdir;

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
    let path = temp_dir.path().join("forward_test_model.bpk");
    let mut save_store = BurnpackStore::from_file(&path);
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
    let mut load_store = BurnpackStore::from_file(&path);
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

#[test]
fn test_store_get_all_snapshots() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save module to bytes
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Get all snapshots (returns &BTreeMap<String, TensorSnapshot>)
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let snapshots = load_store.get_all_snapshots().unwrap();

    // Should have 4 tensors
    assert_eq!(snapshots.len(), 4);

    // Verify tensor names exist (BTreeMap keys)
    assert!(snapshots.contains_key("weight"));
    assert!(snapshots.contains_key("bias"));
    assert!(snapshots.contains_key("nested.gamma"));
    assert!(snapshots.contains_key("nested.beta"));
}

#[test]
fn test_store_get_snapshot_existing() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save module to bytes
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Get a specific snapshot (returns Option<&TensorSnapshot>)
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let snapshot = load_store.get_snapshot("weight").unwrap();

    // Should find the tensor
    assert!(snapshot.is_some());
    let snapshot = snapshot.unwrap();
    assert_eq!(snapshot.full_path(), "weight");
    assert_eq!(snapshot.shape, vec![2, 2]);

    // Verify data can be loaded
    let data = snapshot.to_data().unwrap();
    assert_eq!(data.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_store_get_snapshot_nested() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save module to bytes
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Get a nested snapshot
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let snapshot = load_store.get_snapshot("nested.gamma").unwrap();

    assert!(snapshot.is_some());
    let snapshot = snapshot.unwrap();
    assert_eq!(snapshot.full_path(), "nested.gamma");
    assert_eq!(snapshot.shape, vec![2]);
}

#[test]
fn test_store_get_snapshot_not_found() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save module to bytes
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Try to get a non-existent snapshot
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let snapshot = load_store.get_snapshot("nonexistent").unwrap();

    // Should return None
    assert!(snapshot.is_none());
}

#[test]
fn test_store_keys() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save module to bytes
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Get all keys
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let keys = load_store.keys().unwrap();

    // Should have 4 keys
    assert_eq!(keys.len(), 4);
    assert!(keys.contains(&"weight".to_string()));
    assert!(keys.contains(&"bias".to_string()));
    assert!(keys.contains(&"nested.gamma".to_string()));
    assert!(keys.contains(&"nested.beta".to_string()));
}

#[test]
#[cfg(feature = "std")]
fn test_store_get_all_snapshots_from_file() {
    use tempfile::tempdir;

    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save to file
    let temp_dir = tempdir().unwrap();
    let path = temp_dir.path().join("test_get_all_snapshots.bpk");

    let mut save_store = BurnpackStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    // Get snapshots from file (returns &BTreeMap)
    let mut load_store = BurnpackStore::from_file(&path);
    let snapshots = load_store.get_all_snapshots().unwrap();

    assert_eq!(snapshots.len(), 4);

    // Verify we can load data from a snapshot (use get() on BTreeMap)
    let weight_snapshot = snapshots.get("weight").unwrap();
    let data = weight_snapshot.to_data().unwrap();
    assert_eq!(data.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_store_caching_behavior() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save module to bytes
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Create store and call get_snapshots multiple times
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));

    // First call should populate cache
    let snapshots1 = load_store.get_all_snapshots().unwrap();
    assert_eq!(snapshots1.len(), 4);

    // Second call should return cached data (same reference)
    let snapshots2 = load_store.get_all_snapshots().unwrap();
    assert_eq!(snapshots2.len(), 4);

    // get_snapshot should also use the cache
    let weight = load_store.get_snapshot("weight").unwrap();
    assert!(weight.is_some());
}

#[test]
fn test_store_cache_invalidation_on_save() {
    let device = Default::default();

    // Create first module with specific weights
    let module1 = TestModule::<TestBackend>::new(&device);

    // Save module1 to bytes store
    let mut store = BurnpackStore::from_bytes(None);
    store.collect_from(&module1).unwrap();

    // Populate cache by calling get_snapshots
    let snapshots1 = store.get_all_snapshots().unwrap();
    assert_eq!(snapshots1.len(), 4);
    let weight1_data = snapshots1.get("weight").unwrap().to_data().unwrap();
    let weight1_values: Vec<f32> = weight1_data.to_vec().unwrap();

    // Create a different module with different weights
    let module2 = TestModule::<TestBackend> {
        weight: Param::from_tensor(Tensor::from_data([[10.0, 20.0], [30.0, 40.0]], &device)),
        bias: Param::from_tensor(Tensor::from_data([100.0, 200.0], &device)),
        nested: NestedModule {
            gamma: Param::from_tensor(Tensor::from_data([1000.0, 2000.0], &device)),
            beta: Param::from_tensor(Tensor::from_data([3000.0, 4000.0], &device)),
        },
    };

    // Save module2 - this should invalidate the cache
    store.collect_from(&module2).unwrap();

    // Get snapshots again - should return NEW data, not cached old data
    let snapshots2 = store.get_all_snapshots().unwrap();
    assert_eq!(snapshots2.len(), 4);
    let weight2_data = snapshots2.get("weight").unwrap().to_data().unwrap();
    let weight2_values: Vec<f32> = weight2_data.to_vec().unwrap();

    // Verify the data changed (cache was invalidated)
    assert_ne!(weight1_values, weight2_values);
    assert_eq!(weight2_values, vec![10.0, 20.0, 30.0, 40.0]);
}

/// Test storing and loading quantized weights with BurnpackStore.
/// Regression test for https://github.com/tracel-ai/burn/issues/4179
#[test]
fn test_store_quantized_module_round_trip() {
    use burn_core::module::Quantizer;
    use burn_nn::LinearConfig;
    use burn_tensor::quantization::{
        Calibration, QTensorPrimitive, QuantLevel, QuantParam, QuantValue,
    };

    let device = Default::default();

    // Create a simple linear module (512x512 as in the bug report)
    let linear = LinearConfig::new(512, 512)
        .with_bias(false)
        .init::<TestBackend>(&device);

    // Define quantization scheme (Q8S with tensor-level quantization)
    let scheme = <<TestBackend as burn_tensor::backend::Backend>::QuantizedTensorPrimitive as QTensorPrimitive>::default_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::Tensor)
        .with_param(QuantParam::F32);

    // Quantize the module
    let calibration = Calibration::MinMax;
    let mut quantizer = Quantizer {
        calibration,
        scheme,
    };
    let quantized_linear = linear.quantize_weights(&mut quantizer);

    // Save the quantized module
    let mut save_store = BurnpackStore::from_bytes(None);
    let result = save_store.collect_from(&quantized_linear);
    assert!(
        result.is_ok(),
        "Failed to save quantized module: {:?}",
        result.err()
    );

    // Get the bytes
    let bytes = save_store.get_bytes().expect("Failed to get bytes");

    // Load the bytes and verify we can read the tensor metadata
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let snapshots = load_store
        .get_all_snapshots()
        .expect("Failed to get snapshots");

    // Verify we have the weight tensor
    assert_eq!(snapshots.len(), 1, "Expected 1 tensor (weight)");
    assert!(snapshots.contains_key("weight"), "Expected 'weight' tensor");

    // Verify the tensor metadata
    let weight_snapshot = snapshots.get("weight").unwrap();
    assert_eq!(weight_snapshot.shape, vec![512, 512]);

    // Verify we can load the tensor data
    let weight_data = weight_snapshot
        .to_data()
        .expect("Failed to load tensor data");
    assert_eq!(weight_data.shape, vec![512, 512]);
}

/// Test storing quantized weights with block-level quantization.
#[test]
fn test_store_quantized_module_block_level() {
    use burn_core::module::Quantizer;
    use burn_nn::LinearConfig;
    use burn_tensor::quantization::{
        Calibration, QTensorPrimitive, QuantLevel, QuantParam, QuantValue,
    };

    let device = Default::default();

    // Create a linear module
    let linear = LinearConfig::new(128, 128)
        .with_bias(false)
        .init::<TestBackend>(&device);

    // Define quantization scheme with block-level quantization
    let scheme = <<TestBackend as burn_tensor::backend::Backend>::QuantizedTensorPrimitive as QTensorPrimitive>::default_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([32])) // Block size of 32
        .with_param(QuantParam::F32);

    // Quantize the module
    let calibration = Calibration::MinMax;
    let mut quantizer = Quantizer {
        calibration,
        scheme,
    };
    let quantized_linear = linear.quantize_weights(&mut quantizer);

    // Save the quantized module
    let mut save_store = BurnpackStore::from_bytes(None);
    let result = save_store.collect_from(&quantized_linear);
    assert!(
        result.is_ok(),
        "Failed to save quantized module with block-level quantization: {:?}",
        result.err()
    );

    // Get the bytes and verify round-trip
    let bytes = save_store.get_bytes().expect("Failed to get bytes");

    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let snapshots = load_store
        .get_all_snapshots()
        .expect("Failed to get snapshots");

    assert_eq!(snapshots.len(), 1);
    let weight_snapshot = snapshots.get("weight").unwrap();
    assert_eq!(weight_snapshot.shape, vec![128, 128]);
}
