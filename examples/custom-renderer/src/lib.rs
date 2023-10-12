use burn::data::dataset::source::huggingface::MNISTDataset;
use burn::train::renderer::{MetricState, MetricsRenderer, TrainingProgress};
use burn::train::LearnerBuilder;
use burn::{
    config::Config, data::dataloader::DataLoaderBuilder, optim::AdamConfig,
    tensor::backend::ADBackend,
};
use guide::{data::MNISTBatcher, model::ModelConfig};

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

pub fn run<B: ADBackend>(device: B::Device) {
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
    let batcher_valid = MNISTBatcher::<B::InnerBackend>::new(device.clone());

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::test());

    // artifact dir does not need to be provided when log_to_file is false
    let builder = LearnerBuilder::new("")
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .renderer(CustomRenderer {})
        .log_to_file(false);
    // can be used to interrupt training
    let _interrupter = builder.interrupter();

    let learner = builder.build(model, optim, config.lr);

    let _model_trained = learner.fit(dataloader_train, dataloader_test);
}
