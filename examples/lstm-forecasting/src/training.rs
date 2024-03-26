use crate::{
    dataset::{StockBatcher, StockDataset},
    model::ForecastingModelConfig,
};
use burn::data::dataloader::Dataset;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::{CompactRecorder, NoStdTrainingRecorder};
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::{MetricEarlyStoppingStrategy, StoppingCondition};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{CpuMemory, CpuTemperature, CpuUse, LossMetric},
        LearnerBuilder,
    },
};

static ARTIFACT_DIR: &str = "/tmp/timeseries-forecasting";

#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    pub optimizer: AdamConfig,

    #[config(default = 1)]
    pub input_feature_len: usize,

    #[config(default = 10000)]
    pub dataset_size: usize,

    #[config(default = 20)]
    pub window_size: usize,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // Config
    let optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.98)
        .with_epsilon(1e-9);

    let config = ExpConfig::new(optimizer);

    // Define train/test datasets and dataloaders
    let train_dataset = StockDataset::train(config.window_size, config.dataset_size);
    let test_dataset = StockDataset::test(config.window_size, config.dataset_size);

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Test Dataset Size: {}", test_dataset.len());


    let model = ForecastingModelConfig::new(config.input_feature_len).init(&device);

    let batcher_train = StockBatcher::<B>::new(device.clone());

    let batcher_test = StockBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(test_dataset);

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(model, config.optimizer.init(), 0.0001);

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
