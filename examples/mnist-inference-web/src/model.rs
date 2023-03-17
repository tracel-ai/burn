#![allow(clippy::new_without_default)]

// Orginally copied from the burn/examples/mnist package

use alloc::{format, vec::Vec};

use burn::{
    module::{Module, Param},
    nn::{self, conv::Conv2dPaddingConfig, BatchNorm2d},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Param<ConvBlock<B>>,
    conv2: Param<ConvBlock<B>>,
    conv3: Param<ConvBlock<B>>,
    fc1: Param<nn::Linear<B>>,
    fc2: Param<nn::Linear<B>>,
    activation: nn::GELU,
}

const NUM_CLASSES: usize = 10;

impl<B: Backend> Model<B> {
    pub fn new() -> Self {
        let conv1 = ConvBlock::new([1, 8], [3, 3]); // out: [Batch,8,26,26]
        let conv2 = ConvBlock::new([8, 16], [3, 3]); // out: [Batch,16,24x24]
        let conv3 = ConvBlock::new([16, 24], [3, 3]); // out: [Batch,24,22x22]

        let hidden_size = 24 * 22 * 22;
        let fc1 = nn::Linear::new(&nn::LinearConfig::new(hidden_size, 32).with_bias(false));
        let fc2 = nn::Linear::new(&nn::LinearConfig::new(32, NUM_CLASSES).with_bias(false));

        Self {
            conv1: Param::from(conv1),
            conv2: Param::from(conv2),
            conv3: Param::from(conv3),
            fc1: Param::from(fc1),
            fc2: Param::from(fc2),
            activation: nn::GELU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, heigth, width] = input.dims();

        let x = input.reshape([batch_size, 1, heigth, width]).detach();
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);

        let [batch_size, channels, heigth, width] = x.dims();
        let x = x.reshape([batch_size, channels * heigth * width]);

        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);

        self.fc2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: Param<nn::conv::Conv2d<B>>,
    norm: Param<BatchNorm2d<B>>,
    activation: nn::GELU,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2]) -> Self {
        let conv = nn::conv::Conv2d::new(
            &nn::conv::Conv2dConfig::new(channels, kernel_size)
                .with_padding(Conv2dPaddingConfig::Valid),
        );
        let norm = nn::BatchNorm2d::new(&nn::BatchNorm2dConfig::new(channels[1]));

        Self {
            conv: Param::from(conv),
            norm: Param::from(norm),
            activation: nn::GELU::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);

        self.activation.forward(x)
    }
}
