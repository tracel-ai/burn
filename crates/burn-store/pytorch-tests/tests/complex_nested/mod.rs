use burn::tensor::Tolerance;
use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{
        Device, Tensor,
        activation::{log_softmax, relu},
    },
};
use burn_store::{ModuleSnapshot, PytorchStore};

#[derive(Module, Debug)]
pub struct ConvBlock {
    conv: Conv2d,
    norm: BatchNorm,
}

#[derive(Module, Debug)]
pub struct Net {
    conv_blocks: Vec<ConvBlock>,
    norm1: BatchNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Net {
    pub fn init(device: &Device) -> Self {
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
    pub fn forward(&self, x: Tensor<4>) -> Tensor<2> {
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

impl ConvBlock {
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(x);

        self.norm.forward(x)
    }
}

/// Partial model to test loading of partial records.
#[derive(Module, Debug)]
pub struct PartialNet {
    conv1: ConvBlock,
}

impl PartialNet {
    /// Create a new model from the given record.
    pub fn init(device: &Device) -> Self {
        let conv1 = ConvBlock {
            conv: Conv2dConfig::new([2, 4], [3, 2]).init(device),
            norm: BatchNormConfig::new(4).init(device), // matches conv output channels
        };
        Self { conv1 }
    }

    /// Forward pass of the model.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.conv1.forward(x)
    }
}

/// Model with extra fields to test loading of records (e.g. from a different model).
#[derive(Module, Debug)]
pub struct PartialWithExtraNet {
    conv1: ConvBlock,
    extra_field: bool, // This field is not present in the pytorch model
}

impl PartialWithExtraNet {
    /// Create a new model from the given record.
    pub fn init(device: &Device) -> Self {
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
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.conv1.forward(x)
    }
}

fn model_test(model: Net, precision: f32) {
    let device = Default::default();

    let input = Tensor::<4>::ones([1, 2, 9, 6], &device) - 0.5;

    let output = model.forward(input);

    let expected = Tensor::<2>::from_data(
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

    output
        .to_data()
        .assert_approx_eq::<f32>(&expected.to_data(), Tolerance::absolute(precision));
}

#[test]
fn full_record() {
    let device = Default::default();
    let mut model = Net::init(&device);
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt");
    model
        .load_from(&mut store)
        .expect("Should decode state successfully");

    model_test(model, 1e-8);
}

#[test]
fn full_record_autodiff() {
    let device = Device::default().autodiff();
    let mut model = Net::init(&device);
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt");
    model
        .load_from(&mut store)
        .expect("Should decode state successfully");
}

#[test]
fn half_record() {
    let device = Default::default();
    let mut model = Net::init(&device);
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt");
    model
        .load_from(&mut store)
        .expect("Should decode state successfully");

    model_test(model, 1e-4);
}

#[test]
fn partial_model_loading() {
    let device = Default::default();
    let mut model = PartialNet::init(&device);

    // Load the full model but rename "conv_blocks.0.*" to "conv1.*"
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt")
        .with_key_remapping("conv_blocks\\.0\\.(.*)", "conv1.$1")
        .allow_partial(true);

    model
        .load_from(&mut store)
        .expect("Should decode state successfully");

    let input = Tensor::<4>::ones([1, 2, 9, 6], &device) - 0.5;

    let output = model.forward(input);

    // get the sum of all elements in the output tensor for quick check
    let sum = output.sum();

    assert!((sum.into_scalar::<f64>() - 4.871538).abs() < 0.000002);
}

#[test]
fn extra_field_model_loading() {
    let device = Default::default();
    let mut model = PartialWithExtraNet::init(&device);

    // Load the full model but rename "conv_blocks.0.*" to "conv1.*"
    let mut store = PytorchStore::from_file("tests/complex_nested/complex_nested.pt")
        .with_key_remapping("conv_blocks\\.0\\.(.*)", "conv1.$1")
        .allow_partial(true);

    model
        .load_from(&mut store)
        .expect("Should decode state successfully");

    let input = Tensor::<4>::ones([1, 2, 9, 6], &device) - 0.5;

    let output = model.forward(input);

    // get the sum of all elements in the output tensor for quick check
    let sum = output.sum();

    assert!((sum.into_scalar::<f64>() - 4.871538).abs() < 0.000002);

    assert!(model.extra_field);
}
