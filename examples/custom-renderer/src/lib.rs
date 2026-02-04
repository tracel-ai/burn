use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::AdamConfig,
    tensor::backend::AutodiffBackend,
    train::{
        Learner, SupervisedTraining,
        renderer::{
            EvaluationName, EvaluationProgress, MetricState, MetricsRenderer,
            MetricsRendererEvaluation, MetricsRendererTraining, ProgressType, TrainingProgress,
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

    fn render_train(
        &mut self,
        item: TrainingProgress,
        _progress_indicators: Vec<ProgressType>,
    ) {
        dbg!(item);
    }

    fn render_valid(
        &mut self,
        item: TrainingProgress,
        _progress_indicators: Vec<ProgressType>,
    ) {
        dbg!(item);
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

    fn render_test(
        &mut self,
        item: EvaluationProgress,
        _progress_indicators: Vec<ProgressType>,
    ) {
        dbg!(item);
    }
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // Create the configuration.
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);

    B::seed(&device, config.seed);

    // Create the model and optimizer.
    let model = config.model.init::<B>(&device);
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
