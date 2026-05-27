use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::AdamConfig,
    tensor::Device,
    train::{
        Learner, SupervisedTraining,
        logger::{EvaluationProgressLogger, TrainingProgressLogger},
        renderer::{
            EvaluationName, MetricState, MetricsRenderer, MetricsRendererEvaluation,
            MetricsRendererTraining,
        },
    },
};
use guide::{data::MnistBatcher, model::ModelConfig};

#[derive(Config, Debug)]
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

impl MetricsRendererTraining for CustomRenderer {
    fn update_train(&mut self, _state: MetricState) {}

    fn update_valid(&mut self, _state: MetricState) {}
}

impl TrainingProgressLogger for CustomRenderer {
    fn start(&mut self, _total_epochs: usize, _total_items: Option<usize>) {}

    fn update_epoch(&mut self, _epoch: usize) {}

    fn start_split(&mut self, _split: &str, _total_items: usize) {}

    fn update_split(&mut self, items_processed: usize) {
        dbg!(items_processed);
    }

    fn end_split(&mut self) {}

    fn end(&mut self) {}

    fn log_event_training(&mut self, event: String) {
        dbg!(event);
    }
}

impl MetricsRenderer for CustomRenderer {
    fn manual_close(&mut self) {
        // Nothing to do.
    }

    fn register_metric(&mut self, _definition: burn::train::metric::MetricDefinition) {}
}

impl MetricsRendererEvaluation for CustomRenderer {
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}
}

impl EvaluationProgressLogger for CustomRenderer {
    fn start_global_progress(&mut self, _total_tests: usize) {}

    fn start_test(&mut self, _name: &str, _total_items: usize) {}

    fn update_test_progress(&mut self, items_processed: usize) {
        dbg!(items_processed);
    }

    fn end_test(&mut self) {}

    fn end_global_progress(&mut self) {}

    fn log_event_evaluation(&mut self, event: String) {
        dbg!(event);
    }
}

pub fn run(device: Device) {
    // Create the configuration.
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);

    let device = device.autodiff();
    device.seed(config.seed);

    // Create the model and optimizer.
    let model = config.model.init(&device);
    let optim = config.optimizer.init();

    // Create the batcher.
    let batcher = MnistBatcher::default();

    // Create the dataloaders.
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    // artifact dir does not need to be provided when log_to_file is false
    let training = SupervisedTraining::new("", dataloader_train, dataloader_test)
        .num_epochs(config.num_epochs)
        .renderer(CustomRenderer {})
        .with_application_logger(None);
    // can be used to interrupt training
    let _interrupter = training.interrupter();

    let _model_trained = training.launch(Learner::new(model, optim, config.lr));
}
