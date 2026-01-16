//! Tests for multi-layer model loading with SafeTensors format
use burn_core as burn;

use burn_core::module::Module;
use burn_tensor::{Tensor, backend::Backend};

use burn_nn::{
    BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
    conv::{Conv2d, Conv2dConfig},
};

/// Multi-layer neural network model for testing
#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    conv1: Conv2d<B>,
    norm1: BatchNorm<B>,
    fc1: Linear<B>,
    relu: Relu,
}

impl<B: Backend> Net<B> {
    /// Create a new network instance
    pub fn new(device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([3, 4], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device),
            norm1: BatchNormConfig::new(4).init(device),
            fc1: LinearConfig::new(4 * 8 * 8, 16).init(device),
            relu: Relu::new(),
        }
    }

    /// Forward pass of the model
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.norm1.forward(x);
        let x = self.relu.forward(x);
        // Flatten all dimensions except the batch dimension
        let x = x.flatten(1, 3);
        self.fc1.forward(x)
    }
}

use crate::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use burn_tensor::Tolerance;

type TestBackend = burn_ndarray::NdArray;

/// Path to the multi_layer.safetensors test file
fn get_safetensors_path() -> &'static str {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../burn-import/safetensors-tests/tests/multi_layer/multi_layer.safetensors"
    )
}

#[test]
fn multi_layer_model() {
    let device = Default::default();
    let safetensors_path = get_safetensors_path();

    // Load model from SafeTensors file with PyTorch adapter
    let mut store = SafetensorsStore::from_file(safetensors_path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .validate(false)
        .allow_partial(true);

    let mut model = Net::<TestBackend>::new(&device);
    let result = model.load_from(&mut store).unwrap();

    // Verify loading was successful
    assert!(
        !result.applied.is_empty(),
        "Should have loaded some tensors"
    );
    assert!(
        result.errors.is_empty(),
        "Should have no errors: {:?}",
        result.errors
    );

    // Test forward pass
    let input = Tensor::<TestBackend, 4>::ones([1, 3, 8, 8], &device);
    let output = model.forward(input);

    // Expected output values from PyTorch model
    let expected = Tensor::<TestBackend, 2>::from_data(
        [[
            0.04971555,
            -0.16849735,
            0.05182848,
            -0.18032673,
            0.23138367,
            0.05041867,
            0.13005908,
            -0.32202929,
            -0.07915690,
            -0.03232457,
            -0.19790289,
            -0.17476529,
            -0.19627589,
            -0.21757686,
            -0.31376451,
            0.08377837,
        ]],
        &device,
    );

    // Verify output matches expected values
    output
        .to_data()
        .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::default());
}
