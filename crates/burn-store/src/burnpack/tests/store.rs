#[cfg(feature = "std")]
use crate::KeyRemapper;
use crate::burnpack::store::BurnpackStore;
use crate::{ModuleSnapshot, ModuleSnapshoter, PathFilter};
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
    let path = temp_dir.path().join("test_file_round_trip.burnpack");

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
    let path = temp_dir.path().join("test_model.burnpack");

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
    let path = temp_dir.path().join("test_model_metadata.burnpack");

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

    // Save without extension - should auto-append .burnpack
    let mut save_store = BurnpackStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    // Verify that model.burnpack was created
    let expected_path = temp_dir.path().join("model.burnpack");
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
    let path = temp_dir.path().join("model.burnpack");

    // Save with .burnpack extension - should not double append
    let mut save_store = BurnpackStore::from_file(&path);
    save_store.collect_from(&module).unwrap();

    // Verify that only model.burnpack was created
    assert!(path.exists());
    let double_ext_path = temp_dir.path().join("model.burnpack.burnpack");
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

    // Verify that model.mpk was created (not model.mpk.burnpack)
    assert!(path.exists());
    let burnpack_path = temp_dir.path().join("model.mpk.burnpack");
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
    let burnpack_path = temp_dir.path().join("model.burnpack");
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
    let path = temp_dir.path().join("model.burnpack");

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
    let result = load_module.apply_from(&mut load_store).unwrap();

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
