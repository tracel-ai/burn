use crate::data::MNISTBatch;
use burn::{
    config::Config,
    module::{Forward, Module, Param},
    nn,
    optim::SgdConfig,
    tensor::{
        backend::{ADBackend, Backend},
        loss::cross_entropy_with_logits,
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct MnistConfig {
    #[config(default = 6)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    pub optimizer: SgdConfig,
    pub mlp: MlpConfig,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    mlp: Param<Mlp<B>>,
    input: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
}

#[derive(Config)]
pub struct MlpConfig {
    #[config(default = 3)]
    pub num_layers: usize,
    #[config(default = 0.5)]
    pub dropout: f64,
    #[config(default = 256)]
    pub dim: usize,
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    linears: Param<Vec<nn::Linear<B>>>,
    dropout: nn::Dropout,
    activation: nn::ReLU,
}

impl<B: Backend> Forward<Tensor<B, 2>, Tensor<B, 2>> for Mlp<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        for linear in self.linears.iter() {
            x = linear.forward(x);
            x = self.dropout.forward(x);
            x = self.activation.forward(x);
        }

        x
    }
}

impl<B: Backend> Forward<Tensor<B, 2>, Tensor<B, 2>> for Model<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        x = self.input.forward(x);
        x = self.mlp.forward(x);
        x = self.output.forward(x);

        x
    }
}

impl<B: Backend> Forward<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn forward(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = cross_entropy_with_logits(&output, &targets);

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: ADBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward(item);
        TrainOutput::new(item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward(item)
    }
}

impl<B: Backend> Mlp<B> {
    pub fn new(config: &MlpConfig) -> Self {
        let mut linears = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            let linear = nn::Linear::new(&nn::LinearConfig::new(config.dim, config.dim));
            linears.push(linear);
        }

        Self {
            linears: Param::new(linears),
            dropout: nn::Dropout::new(&nn::DropoutConfig::new(0.3)),
            activation: nn::ReLU::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn new(config: &MnistConfig, d_input: usize, num_classes: usize) -> Self {
        let mlp = Mlp::new(&config.mlp);
        let output = nn::Linear::new(&nn::LinearConfig::new(config.mlp.dim, num_classes));
        let input = nn::Linear::new(&nn::LinearConfig::new(d_input, config.mlp.dim));

        Self {
            mlp: Param::new(mlp),
            output: Param::new(output),
            input: Param::new(input),
        }
    }
}
