use crate::dataset::{DiabetesBatcher, DiabetesDataset};
use crate::model::RegressionModelConfig;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    optim::SgdConfig,
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::store::{Aggregate, Direction, Split},
        metric::LossMetric,
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

static ARTIFACT_DIR: &str = "/tmp/burn-example-regression";

#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: SgdConfig,

    #[config(default = 10)]
    pub input_feature_len: usize,

    #[config(default = 442)]
    pub dataset_size: usize,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // Config
    let optimizer = SgdConfig::new();
    let config = ExpConfig::new(optimizer);
    let model = RegressionModelConfig::new(config.input_feature_len).init(&device);
    B::seed(config.seed);

    // Define train/test datasets and dataloaders

    let train_dataset = DiabetesDataset::train();
    let test_dataset = DiabetesDataset::test();

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Test Dataset Size: {}", test_dataset.len());

    let batcher_train = DiabetesBatcher::<B>::new(device.clone());

    let batcher_test = DiabetesBatcher::<B::InnerBackend>::new(device.clone());

    // Since dataset size is small, we do full batch gradient descent and set batch size equivalent to size of dataset

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(train_dataset.len())
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(test_dataset.len())
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), 5e-3);

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
