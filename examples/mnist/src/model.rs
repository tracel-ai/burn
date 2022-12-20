use crate::{
    data::MNISTBatch,
    mlp::{Mlp, MlpConfig},
};
use burn::{
    config::Config,
    module::{Module, Param},
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
    #[config(default = 12)]
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

impl<B: Backend> Model<B> {
    pub fn new(config: &MnistConfig, d_input: usize, num_classes: usize) -> Self {
        let mlp = Mlp::new(&config.mlp);
        let output = nn::Linear::new(&nn::LinearConfig::new(config.mlp.d_model, num_classes));
        let input = nn::Linear::new(&nn::LinearConfig::new(d_input, config.mlp.d_model));

        Self {
            mlp: Param::new(mlp),
            output: Param::new(output),
            input: Param::new(input),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        x = self.input.forward(x);
        x = self.mlp.forward(x);
        x = self.output.forward(x);

        x
    }

    pub fn forward_classification(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
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

impl<B: ADBackend> TrainStep<B, MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> TrainOutput<B, ClassificationOutput<B>> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
