// Test Burnpack storage in no-std environment

use burn::{
    module::Module,
    nn,
    tensor::{Tensor, backend::Backend},
};

use burn_store::{BurnpackStore, ModuleSnapshot, PathFilter};

/// Simple model for testing Burnpack storage
#[derive(Module, Debug)]
pub struct TestModel<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
    batch_norm: nn::BatchNorm<B>,
}

impl<B: Backend> TestModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            linear1: nn::LinearConfig::new(10, 20).init(device),
            linear2: nn::LinearConfig::new(20, 10).init(device),
            batch_norm: nn::BatchNormConfig::new(10).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.linear2.forward(x);
        // Apply batch norm (expand to 3D, apply, then squeeze back)
        let x: Tensor<B, 3> = x.unsqueeze_dim(2);
        let x = self.batch_norm.forward(x);
        x.squeeze_dim(2)
    }
}

/// Test basic Burnpack save and load in no-std
pub fn test_burnpack_basic<B: Backend>(device: &B::Device) {
    // Create a model
    let model = TestModel::<B>::new(device);

    // Save to bytes (no file I/O in no-std)
    let mut save_store = BurnpackStore::from_bytes(None);
    model
        .collect_to(&mut save_store)
        .expect("Failed to save model");

    // Get the serialized bytes
    let bytes = save_store.get_bytes().expect("Failed to get bytes");

    // Load from bytes
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut loaded_model = TestModel::<B>::new(device);
    let result = loaded_model
        .apply_from(&mut load_store)
        .expect("Failed to load model");

    // Verify all tensors were loaded
    assert!(result.is_success(), "Should have no errors");
    assert!(!result.applied.is_empty(), "Should have loaded tensors");

    // Test that the model still works
    let input = Tensor::<B, 2>::ones([2, 10], device);
    let _output = loaded_model.forward(input);
}

/// Test Burnpack with filtering in no-std
pub fn test_burnpack_filtering<B: Backend>(device: &B::Device) {
    let model = TestModel::<B>::new(device);

    // Save only linear1 weights
    let filter = PathFilter::new()
        .with_full_path("linear1.weight")
        .with_full_path("linear1.bias");
    let mut save_store = BurnpackStore::from_bytes(None).with_filter(filter);
    model
        .collect_to(&mut save_store)
        .expect("Failed to save filtered model");

    let bytes = save_store.get_bytes().expect("Failed to get bytes");

    // Load with partial loading allowed
    let mut load_store = BurnpackStore::from_bytes(Some(bytes)).allow_partial(true);
    let mut partial_model = TestModel::<B>::new(device);
    let result = partial_model
        .apply_from(&mut load_store)
        .expect("Failed to load partial model");

    // Verify that only linear1 was loaded
    assert_eq!(result.applied.len(), 2, "Should have loaded 2 tensors");
    assert!(!result.missing.is_empty(), "Should have missing tensors");
}

/// Test Burnpack with metadata in no-std
pub fn test_burnpack_metadata<B: Backend>(device: &B::Device) {
    let model = TestModel::<B>::new(device);

    // Save with metadata
    let mut save_store = BurnpackStore::from_bytes(None)
        .metadata("version", "1.0.0")
        .metadata("environment", "no-std")
        .metadata("model_type", "test");
    model
        .collect_to(&mut save_store)
        .expect("Failed to save model with metadata");

    let bytes = save_store.get_bytes().expect("Failed to get bytes");

    // Load and verify it works
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut loaded_model = TestModel::<B>::new(device);
    let result = loaded_model
        .apply_from(&mut load_store)
        .expect("Failed to load model with metadata");

    assert!(result.is_success(), "Should load successfully");
}

// Note: Key remapping test is omitted as KeyRemapper requires std feature

// Note: Regex filtering test is omitted as with_regex requires std feature

/// Test Burnpack with match_all in no-std
pub fn test_burnpack_match_all<B: Backend>(device: &B::Device) {
    let model = TestModel::<B>::new(device);

    // Save with match_all (should save everything)
    let mut save_store = BurnpackStore::from_bytes(None).match_all();
    model
        .collect_to(&mut save_store)
        .expect("Failed to save model");

    let bytes = save_store.get_bytes().expect("Failed to get bytes");

    // Load everything
    let mut load_store = BurnpackStore::from_bytes(Some(bytes));
    let mut loaded_model = TestModel::<B>::new(device);
    let result = loaded_model
        .apply_from(&mut load_store)
        .expect("Failed to load model");

    assert!(result.is_success(), "Should load successfully");
    // linear1 (weight, bias) + linear2 (weight, bias) + batch_norm (4 params)
    assert_eq!(result.applied.len(), 8, "Should load all 8 tensors");
    assert!(result.missing.is_empty(), "Should have no missing tensors");
    assert!(result.unused.is_empty(), "Should have no unused tensors");
}

/// Run all Burnpack no-std tests
pub fn run_all_tests<B: Backend>(device: &B::Device) {
    test_burnpack_basic::<B>(device);
    test_burnpack_filtering::<B>(device);
    test_burnpack_metadata::<B>(device);
    // test_burnpack_remapping requires KeyRemapper which needs std
    // test_burnpack_regex_filter requires with_regex which needs std
    test_burnpack_match_all::<B>(device);
}