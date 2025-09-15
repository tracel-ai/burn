// Test SafeTensors storage in no-std environment

use burn::{
    module::Module,
    nn,
    tensor::{Tensor, backend::Backend},
};

use burn_store::{ModuleSnapshot, SafetensorsStore};

/// Simple model for testing SafeTensors storage
#[derive(Module, Debug)]
pub struct TestModel<B: Backend> {
    linear1: nn::Linear<B>,
    linear2: nn::Linear<B>,
}

impl<B: Backend> TestModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            linear1: nn::LinearConfig::new(10, 20).init(device),
            linear2: nn::LinearConfig::new(20, 10).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        self.linear2.forward(x)
    }
}

/// Test basic SafeTensors save and load in no-std
pub fn test_safetensors_basic<B: Backend>(device: &B::Device) {
    // Create a model
    let model = TestModel::<B>::new(device);

    // Save to bytes (no file I/O in no-std)
    let mut save_store = SafetensorsStore::from_bytes(None);
    model
        .collect_to(&mut save_store)
        .expect("Failed to save model");

    // Get the serialized bytes
    let bytes = save_store.get_bytes().expect("Failed to get bytes");

    // Load from bytes
    let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
    let mut loaded_model = TestModel::<B>::new(device);
    loaded_model
        .apply_from(&mut load_store)
        .expect("Failed to load model");

    // Test that the model still works
    let input = Tensor::<B, 2>::ones([2, 10], device);
    let _output = loaded_model.forward(input);
}

/// Test SafeTensors with filtering in no-std
pub fn test_safetensors_filtering<B: Backend>(device: &B::Device) {
    let model = TestModel::<B>::new(device);

    // Save only linear1 weights
    let mut save_store = SafetensorsStore::from_bytes(None)
        .with_full_path("linear1.weight")
        .with_full_path("linear1.bias");
    model
        .collect_to(&mut save_store)
        .expect("Failed to save filtered model");

    let bytes = save_store.get_bytes().expect("Failed to get bytes");

    // Load with partial loading allowed
    let mut load_store = SafetensorsStore::from_bytes(Some(bytes)).allow_partial(true);
    let mut partial_model = TestModel::<B>::new(device);
    let result = partial_model
        .apply_from(&mut load_store)
        .expect("Failed to load partial model");

    // Verify that only linear1 was loaded
    assert_eq!(result.applied.len(), 2, "Should have loaded 2 tensors");
    assert!(!result.missing.is_empty(), "Should have missing tensors");
}

/// Test SafeTensors with metadata in no-std
pub fn test_safetensors_metadata<B: Backend>(device: &B::Device) {
    let model = TestModel::<B>::new(device);

    // Save with metadata
    let mut save_store = SafetensorsStore::from_bytes(None)
        .metadata("version", "1.0.0")
        .metadata("environment", "no-std");
    model
        .collect_to(&mut save_store)
        .expect("Failed to save model with metadata");

    let bytes = save_store.get_bytes().expect("Failed to get bytes");

    // Load and verify it works
    let mut load_store = SafetensorsStore::from_bytes(Some(bytes));
    let mut loaded_model = TestModel::<B>::new(device);
    loaded_model
        .apply_from(&mut load_store)
        .expect("Failed to load model with metadata");
}

/// Run all SafeTensors no-std tests
pub fn run_all_tests<B: Backend>(device: &B::Device) {
    test_safetensors_basic::<B>(device);
    test_safetensors_filtering::<B>(device);
    test_safetensors_metadata::<B>(device);
}
