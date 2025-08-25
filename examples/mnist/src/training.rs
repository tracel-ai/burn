use std::time::Duration;

use crate::{data::MnistBatcher, model::Model};

use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        EvaluatorBuilder, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric,
            store::{Aggregate, Direction, Split},
        },
        renderer::MetricsRenderer,
    },
};

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 30)]
    pub num_epochs: usize,

    #[config(default = 256)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    create_artifact_dir(ARTIFACT_DIR);
    // Config
    let config_optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));

    let config = MnistTrainingConfig::new(config_optimizer);
    B::seed(config.seed);

    let model = Model::<B>::new(&device);

    let dataloader_train = DataLoaderBuilder::new(MnistBatcher::new(true))
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());
    let dataloader_valid = DataLoaderBuilder::new(MnistBatcher::new(true))
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 5 },
        ))
        .num_epochs(config.num_epochs)
        // .summary()
        .learning_strategy(burn::train::LearningStrategy::SingleDevice(device))
        .build(model, config.optimizer.init(), 1.0e-3);

    let result = learner.fit(dataloader_train, dataloader_valid);

    result
        .model
        .clone()
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    let evaluate =
        |augmentation: bool, model: Model<B::InnerBackend>, renderer: Box<dyn MetricsRenderer>| {
            let batcher = MnistBatcher::new(augmentation);
            let name = match augmentation {
                true => "MNIST-Augmented",
                false => "MNIST-Plain",
            };
            let dataloader_test = DataLoaderBuilder::new(batcher)
                .batch_size(config.batch_size)
                .num_workers(config.num_workers)
                .build(MnistDataset::test());

            let evaluator = EvaluatorBuilder::new(ARTIFACT_DIR)
                .renderer(renderer)
                .metric_numeric(AccuracyMetric::new())
                .metric_numeric(LossMetric::new())
                .build(model);

            evaluator.eval(name, dataloader_test)
        };

    let renderer = result.renderer;
    let rendered = evaluate(false, result.model.clone(), renderer);
    let mut rendered = evaluate(true, result.model, rendered);

    rendered.manual_close();
}
