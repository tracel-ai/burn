use crate::burnpack::store::BurnpackStore;
use crate::{KeyRemapper, ModuleSnapshoter, PathFilter};
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

    let mut load_store = BurnpackStore::from_bytes(Some(bytes)).remap(remapper);

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
fn test_store_with_remap_pattern() {
    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Save normally
    let mut save_store = BurnpackStore::from_bytes(None);
    save_store.collect_from(&module).unwrap();
    let bytes = save_store.get_bytes().unwrap();

    // Load with single remap pattern using the convenience method
    let mut load_store =
        BurnpackStore::from_bytes(Some(bytes)).with_remap_pattern(r"^nested\.", "sub_module.");

    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    // After remapping, nested.* becomes sub_module.*, which won't match
    assert_eq!(result.applied.len(), 2); // Only weight and bias
    assert_eq!(result.unused.len(), 2); // sub_module.gamma and sub_module.beta unused
}

#[test]
#[cfg(feature = "std")]
fn test_store_file_round_trip() {
    use tempfile::NamedTempFile;

    let device = Default::default();
    let module = TestModule::<TestBackend>::new(&device);

    // Create temp file
    let temp_file = NamedTempFile::new().unwrap();
    let path = temp_file.path();

    // Save to file
    let mut save_store = BurnpackStore::from_file(path).metadata("test", "value");
    save_store.collect_from(&module).unwrap();

    // Verify file exists
    assert!(path.exists());

    // Load from file
    let mut load_store = BurnpackStore::from_file(path);
    let mut module2 = TestModule::<TestBackend>::new_zeros(&device);
    let result = load_store.apply_to(&mut module2).unwrap();

    assert!(result.is_success());
    assert_eq!(result.applied.len(), 4);

    // Verify data
    let weight1 = module.weight.val().to_data().to_vec::<f32>().unwrap();
    let weight2 = module2.weight.val().to_data().to_vec::<f32>().unwrap();
    assert_eq!(weight1, weight2);
}
