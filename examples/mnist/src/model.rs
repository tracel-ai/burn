use crate::{
    conv::{ConvBlock, ConvBlockConfig},
    data::MNISTBatch,
    mlp::{Mlp, MlpConfig},
};
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{self, loss::CrossEntropyLoss},
    optim::AdamConfig,
    tensor::{
        backend::{ADBackend, Backend},
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct MnistConfig {
    #[config(default = 6)]
    pub num_epochs: usize,
    #[config(default = 12)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    pub optimizer: AdamConfig,
    pub mlp: MlpConfig,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    mlp: Param<Mlp<B>>,
    conv: Param<ConvBlock<B>>,
    input: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
    num_classes: usize,
}

impl<B: Backend> Model<B> {
    pub fn new(config: &MnistConfig, d_input: usize, num_classes: usize) -> Self {
        let mlp = Mlp::new(&config.mlp);
        let output = nn::Linear::new(&nn::LinearConfig::new(config.mlp.d_model, num_classes));
        let input = nn::Linear::new(&nn::LinearConfig::new(d_input, config.mlp.d_model));
        let conv = ConvBlock::new(&ConvBlockConfig::new([1, 1]));

        Self {
            mlp: Param::from(mlp),
            conv: Param::from(conv),
            output: Param::from(output),
            input: Param::from(input),
            num_classes,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, heigth, width] = input.dims();

        let x = input.reshape([batch_size, 1, heigth, width]).detach();
        let x = self.conv.forward(x);
        let x = x.reshape([batch_size, heigth * width]);

        let x = self.input.forward(x);
        let x = self.mlp.forward(x);

        self.output.forward(x)
    }

    pub fn forward_classification(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = CrossEntropyLoss::new(None);
        let loss = loss.forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: ADBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
