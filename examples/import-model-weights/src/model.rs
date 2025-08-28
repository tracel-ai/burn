use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::{log_softmax, relu},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    norm1: BatchNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    norm2: BatchNorm<B>,
}

impl<B: Backend> Model<B> {
    pub fn init(device: &B::Device) -> Self {
        let conv1 = Conv2dConfig::new([1, 8], [3, 3]).init(device);
        let conv2 = Conv2dConfig::new([8, 16], [3, 3]).init(device);
        let conv3 = Conv2dConfig::new([16, 24], [3, 3]).init(device);
        let norm1 = BatchNormConfig::new(24).init(device);
        let fc1 = LinearConfig::new(11616, 32).init(device);
        let fc2 = LinearConfig::new(32, 10).init(device);
        let norm2 = BatchNormConfig::new(10).init(device);

        Self {
            conv1,
            conv2,
            conv3,
            norm1,
            fc1,
            fc2,
            norm2,
        }
    }

    pub fn forward(&self, input1: Tensor<B, 4>) -> Tensor<B, 2> {
        let conv1_out1 = self.conv1.forward(input1);
        let relu1_out1 = relu(conv1_out1);
        let conv2_out1 = self.conv2.forward(relu1_out1);
        let relu2_out1 = relu(conv2_out1);
        let conv3_out1 = self.conv3.forward(relu2_out1);
        let relu3_out1 = relu(conv3_out1);
        let norm1_out1 = self.norm1.forward(relu3_out1);
        let flatten1_out1 = norm1_out1.flatten(1, 3);
        let fc1_out1 = self.fc1.forward(flatten1_out1);
        let relu4_out1 = relu(fc1_out1);
        let fc2_out1 = self.fc2.forward(relu4_out1);
        let norm2_out1 = self.norm2.forward(fc2_out1);
        log_softmax(norm2_out1, 1)
    }
}
