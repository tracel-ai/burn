// Generated from tests/data/model1/model1.onnx by burn-import


use burn::tensor::activation::log_softmax;
use burn::tensor::activation::relu;
use burn::{
    module::Module,
    nn,
    record::{DefaultRecordSettings, Record},
    tensor::{backend::Backend, Tensor},
};

pub const INPUT1_SHAPE: [usize; 4usize] = [1usize, 1usize, 8usize, 8usize];
pub const OUTPUT1_SHAPE: [usize; 2usize] = [1usize, 288usize];

///This is a generated model from an ONNX file
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv2d1: nn::conv::Conv2d<B>,
    batchnormalization1: nn::BatchNorm<B, 2>,
    linear1: nn::Linear<B>,
    batchnormalization2: nn::BatchNorm<B, 0>,
}

#[allow(dead_code)]
#[allow(clippy::new_without_default)]
#[allow(clippy::let_and_return)]
impl<B: Backend> Model<B> {
    pub fn new() -> Self {
        let conv2d1 = nn::conv::Conv2dConfig::new([1usize, 8usize], [3usize, 3usize])
            .with_padding(nn::conv::Conv2dPaddingConfig::Valid)
            .with_bias(true)
            .init();
        let batchnormalization1 = nn::BatchNormConfig::new(8usize)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init();
        let linear1 = nn::LinearConfig::new(288usize, 10usize)
            .with_bias(true)
            .init();
        let batchnormalization2 = nn::BatchNormConfig::new(10usize)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init();
        Self {
            conv2d1,
            batchnormalization1,
            linear1,
            batchnormalization2,
        }
    }
    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
        let conv2d1_out1 = self.conv2d1.forward(input1);
        let relu1_out1 = relu(conv2d1_out1);
        let batchnormalization1_out1 = self.batchnormalization1.forward(relu1_out1);
        let flatten1_out1 = batchnormalization1_out1.flatten(1usize, 3usize);
        let linear1_out1 = self.linear1.forward(flatten1_out1);
        let batchnormalization2_out1 = self.batchnormalization2.forward(linear1_out1);
        let logsoftmax1_out1 = log_softmax(batchnormalization2_out1, 1usize);
        logsoftmax1_out1
    }
    pub fn load_state(self) -> Self {
        let record = Record::load::<DefaultRecordSettings>("./model1".into()).unwrap();
        self.load_record(record)
    }
}
