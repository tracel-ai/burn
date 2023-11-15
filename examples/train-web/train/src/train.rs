use crate::{data::MNISTBatcher, mnist::MNISTDataset};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig, ReLU,
    },
    optim::AdamConfig,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        renderer::{MetricState, MetricsRenderer, TrainingProgress},
        ClassificationOutput, LearnerBuilder,
    },
};

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
}

struct CustomRenderer {}

impl MetricsRenderer for CustomRenderer {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}

    fn render_train(&mut self, item: TrainingProgress) {
        dbg!(item);
    }

    fn render_valid(&mut self, item: TrainingProgress) {
        dbg!(item);
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

pub fn run<B: AutodiffBackend>(device: B::Device, labels: &[u8], images: &[u8], lengths: &[u16]) {
    // Create the configuration.
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);

    B::seed(config.seed);

    // Create the model and optimizer.
    let model = config.model.init();
    let optim = config.optimizer.init();

    // Create the batcher.
    let batcher_train = MNISTBatcher::<B>::new(device.clone());
    // let batcher_valid = MNISTBatcher::<B::InnerBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::new(labels, images, lengths));

    // let dataloader_test = DataLoaderBuilder::new(batcher_valid)
    //     .batch_size(config.batch_size)
    //     .shuffle(config.seed)
    //     .num_workers(config.num_workers)
    //     .build(MNISTDataset::test());

    // artifact dir does not need to be provided when log_to_file is false
    let builder: LearnerBuilder<
        B,
        ClassificationOutput<B>,
        ClassificationOutput<<B>::InnerBackend>,
        _,
        _,
        _,
    > = LearnerBuilder::new("")
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .renderer(CustomRenderer {})
        .log_to_file(false);
    // can be used to interrupt training
    let _interrupter = builder.interrupter();

    let _learner = builder.build(model, optim, config.lr);

    // let _model_trained = learner.fit(dataloader_train, dataloader_test);
}
