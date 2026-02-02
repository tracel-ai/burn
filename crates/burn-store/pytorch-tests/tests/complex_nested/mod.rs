use burn::tensor::Tolerance;
use burn::tensor::ops::FloatElem;
use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        Tensor,
        activation::{log_softmax, relu},
        backend::Backend,
    },
};
use burn_autodiff::Autodiff;
use burn_store::{ModuleSnapshot, PytorchStore};

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B>,
}

#[derive(Module, Debug)]
pub struct Net<B: Backend> {
    conv_blocks: Vec<ConvBlock<B>>,
    norm1: BatchNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> Net<B> {
    pub fn init(device: &B::Device) -> Self {
        let conv_blocks = vec![
            ConvBlock {
                conv: Conv2dConfig::new([2, 4], [3, 2]).init(device),
                norm: BatchNormConfig::new(4).init(device), // matches conv output channels
            },
            ConvBlock {
                conv: Conv2dConfig::new([4, 6], [3, 2]).init(device),
                norm: BatchNormConfig::new(6).init(device), // matches conv output channels
            },
        ];
        let norm1 = BatchNormConfig::new(6).init(device);
        let fc1 = LinearConfig::new(120, 12).init(device);
        let fc2 = LinearConfig::new(12, 10).init(device);

        Self {
            conv_blocks,
            norm1,
            fc1,
            fc2,
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv_blocks[0].forward(x);
        let x = self.conv_blocks[1].forward(x);
        let x = self.norm1.forward(x);
        let x = x.reshape([0, -1]);
        let x = self.fc1.forward(x);
        let x = relu(x);
        let x = self.fc2.forward(x);

        log_softmax(x, 1)
    }
}

impl<B: Backend> ConvBlock<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);

        self.norm.forward(x)
    }
}

/// Partial model to test loading of partial records.
#[derive(Module, Debug)]
pub struct PartialNet<B: Backend> {
    conv1: ConvBlock<B>,
}

impl<B: Backend> PartialNet<B> {
    /// Create a new model from the given record.
    pub fn init(device: &B::Device) -> Self {
        let conv1 = ConvBlock {
            conv: Conv2dConfig::new([2, 4], [3, 2]).init(device),
            norm: BatchNormConfig::new(4).init(device), // matches conv output channels
        };
        Self { conv1 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv1.forward(x)
    }
}

/// Model with extra fields to test loading of records (e.g. from a different model).
#[derive(Module, Debug)]
pub struct PartialWithExtraNet<B: Backend> {
    conv1: ConvBlock<B>,
    extra_field: bool, // This field is not present in the pytorch model
}

impl<B: Backend> PartialWithExtraNet<B> {
    /// Create a new model from the given record.
    pub fn init(device: &B::Device) -> Self {
        let conv1 = ConvBlock {
            conv: Conv2dConfig::new([2, 4], [3, 2]).init(device),
            norm: BatchNormConfig::new(4).init(device), // matches conv output channels
        };

        Self {
            conv1,
            extra_field: true,
        }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv1.forward(x)
    }
}

type TestBackend = burn_ndarray::NdArray<f32>;

fn model_test(model: Net<TestBackend>, precision: f32) {
    let device = Default::default();

    let input = Tensor::<TestBackend, 4>::ones([1, 2, 9, 6], &device) - 0.5;

    let output = model.forward(input);

    let expected = Tensor::<TestBackend, 2>::from_data(
        [[
            -2.306_613,
            -2.058_945_4,
            -2.298_372_7,
            -2.358_294,
            -2.296_395_5,
            -2.416_090_5,
            -2.107_669,
            -2.428_420_8,
            -2.526_469,
            -2.319_918_6,
        ]],
        &device,
    );

    output.to_data().assert_approx_eq::<FloatElem<TestBackend>>(
        &expected.to_data(),
        Tolerance::absolute(precision),
    );
}

#[test]
fn full_record() {
    let device = Default::default();
    let mut model = Net::<TestBackend>::init(&device);
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt");
    model
        .load_from(&mut store)
        .expect("Should decode state successfully");

    model_test(model, 1e-8);
}

#[test]
fn full_record_autodiff() {
    let device = Default::default();
    let mut model = Net::<Autodiff<TestBackend>>::init(&device);
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt");
    model
        .load_from(&mut store)
        .expect("Should decode state successfully");
}

#[test]
fn half_record() {
    let device = Default::default();
    let mut model = Net::<TestBackend>::init(&device);
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt");
    model
        .load_from(&mut store)
        .expect("Should decode state successfully");

    model_test(model, 1e-4);
}

#[test]
fn partial_model_loading() {
    let device = Default::default();
    let mut model = PartialNet::<TestBackend>::init(&device);

    // Load the full model but rename "conv_blocks.0.*" to "conv1.*"
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt")
        .with_key_remapping("conv_blocks\\.0\\.(.*)", "conv1.$1")
        .allow_partial(true);

    model
        .load_from(&mut store)
        .expect("Should decode state successfully");

    let input = Tensor::<TestBackend, 4>::ones([1, 2, 9, 6], &device) - 0.5;

    let output = model.forward(input);

    // get the sum of all elements in the output tensor for quick check
    let sum = output.sum();

    assert!((sum.into_scalar() - 4.871538).abs() < 0.000002);
}

#[test]
fn extra_field_model_loading() {
    let device = Default::default();
    let mut model = PartialWithExtraNet::<TestBackend>::init(&device);

    // Load the full model but rename "conv_blocks.0.*" to "conv1.*"
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt")
        .with_key_remapping("conv_blocks\\.0\\.(.*)", "conv1.$1")
        .allow_partial(true);

    model
        .load_from(&mut store)
        .expect("Should decode state successfully");

    let input = Tensor::<TestBackend, 4>::ones([1, 2, 9, 6], &device) - 0.5;

    let output = model.forward(input);

    // get the sum of all elements in the output tensor for quick check
    let sum = output.sum();

    assert!((sum.into_scalar() - 4.871538).abs() < 0.000002);

    assert!(model.extra_field);
}
