use crate::{data::MnistBatcher, model::Model};

use burn::{
    collective::{AllReduceStrategy, CollectiveConfig},
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

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

    // Data
    let batcher = MnistBatcher::default();

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

    // for collective ops
    let collective =
        CollectiveConfig::default().with_local_all_reduce_strategy(AllReduceStrategy::Tree(3));

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
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .learning_strategy(burn::train::ddp(vec![device], collective))
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}
