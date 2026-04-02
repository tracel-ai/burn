use crate::data::MnistBatch;
use burn::{
    nn::{
        loss::CrossEntropyLossConfig,
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, Initializer, PaddingConfig2d,
    },
    prelude::*,
    tensor::{backend::AutodiffBackend, Distribution},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};
use rand::rngs::{StdRng, SysRng};
use rand::SeedableRng;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    dropout: nn::Dropout,
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
    fc4: nn::Linear<B>,
    fc5: nn::Linear<B>,
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
        let conv1 = ConvBlock::new([1, 64], [3, 3], device, true); // out: max_pool -> [Batch,32,13,13]
        let conv2 = ConvBlock::new([64, 64], [3, 3], device, true); // out: max_pool -> [Batch,64,5,5]
                                                                    // let hidden_size = 64 * 5 * 5;
        let a = 4096;
        let fc1 = nn::LinearConfig::new(28 * 28, a).init(device);
        let fc2 = nn::LinearConfig::new(a, a).init(device);
        let fc3 = nn::LinearConfig::new(a, a).init(device);
        let fc4 = nn::LinearConfig::new(a, a).init(device);
        let fc5 = nn::LinearConfig::new(a, NUM_CLASSES).init(device);

        let dropout = nn::DropoutConfig::new(0.25).init();

        Self {
            conv1,
            conv2,
            dropout,
            fc1,
            fc2,
            fc3,
            fc4,
            fc5,
            activation: nn::Gelu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();

        // let x = input.reshape([batch_size, 1, height, width]).detach();
        // let x = self.conv1.forward(x);
        // let x = self.conv2.forward(x);

        // let [batch_size, channels, height, width] = x.dims();
        // let x = x.reshape([batch_size, channels * height * width]);

        // let x = self.fc1.forward(x);
        // let x = self.activation.forward(x);
        // let x = self.dropout.forward(x);

        // let x = self.fc2.forward(x);
        // let x = self.activation.forward(x);
        // let x = self.dropout.forward(x);

        // self.fc3.forward(x)

        ///////////////////////////////
        // let data = TensorData::random::<f32, _, _>(
        //     [1, 128],
        //     Distribution::Default,
        //     &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
        // );
        // let x = Tensor::from_data(data, &self.fc3.weight.device());
        // let x = x.repeat_dim(0, input.shape()[0]);
        // let x = self.fc3.forward(x);

        //////////////////////////////
        let x = input.reshape([batch_size, height * width]).detach();

        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);

        let x = self.fc2.forward(x);
        let x = self.activation.forward(x);

        self.fc3.forward(x)
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
    norm: BatchNorm<B>,
    pool: Option<MaxPool2d>,
    activation: nn::Relu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(
        channels: [usize; 2],
        kernel_size: [usize; 2],
        device: &B::Device,
        pool: bool,
    ) -> Self {
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

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: MnistBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        let out = TrainOutput::new(self, item.loss.backward(), item);

        out
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = MnistBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
