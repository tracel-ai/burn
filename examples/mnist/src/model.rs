use crate::data::MnistBatch;
use burn::{
    nn::{
        BatchNorm, PaddingConfig2d,
        loss::CrossEntropyLossConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    dropout: nn::Dropout,
    linears: Vec<nn::Linear<B>>,
    head: nn::Linear<B>,
    activation: nn::Gelu,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(&device)
    }
}

const NUM_CLASSES: usize = 10;

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = ConvBlock::new([1, 64], [3, 3], device, false); // out: [Batch,32,26,26]
        let conv2 = ConvBlock::new([64, 64], [3, 3], device, false); // out: [Batch,64,24,24]
        let hidden_size = 64 * 24 * 24;
        let mut linears = vec![nn::LinearConfig::new(hidden_size, 64).init(device)];

        for _ in 0..1 {
            linears.push(nn::LinearConfig::new(64, 64).init(device));
        }
        let head = nn::LinearConfig::new(64, NUM_CLASSES).init(device);

        let dropout = nn::DropoutConfig::new(0.25).init();

        Self {
            conv1,
            conv2,
            dropout,
            linears,
            head,
            activation: nn::Gelu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();

        let x = input.reshape([batch_size, 1, height, width]).detach();
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);

        let [batch_size, channels, height, width] = x.dims();
        let mut x = x.reshape([batch_size, channels * height * width]);

        for linear in self.linears.iter() {
            x = linear.forward(x);
            x = self.activation.forward(x);
            x = self.dropout.forward(x);
        }

        self.head.forward(x)
    }

    pub fn forward_classification(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: nn::conv::Conv2d<B>,
    norm: BatchNorm<B, 2>,
    pool: Option<MaxPool2d>,
    activation: nn::Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device, pool: bool) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Valid)
            .init(device);
        let norm = nn::BatchNormConfig::new(channels[1]).init(device);
        let pool = if pool {
            Some(MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init())
        } else {
            None
        };

        Self {
            conv,
            norm,
            pool,
            activation: nn::Relu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);
        let x = self.activation.forward(x);

        if let Some(pool) = &self.pool {
            pool.forward(x)
        } else {
            x
        }
    }
}

impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
