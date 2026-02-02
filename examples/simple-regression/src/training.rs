use crate::dataset::{HousingBatcher, HousingDataset};
use crate::model::RegressionModelConfig;
use burn::optim::AdamConfig;
use burn::train::{Learner, SupervisedTraining};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::metric::LossMetric,
};

#[derive(Config, Debug)]
pub struct ExpConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    #[config(default = 1337)]
    pub seed: u64,

    pub optimizer: AdamConfig,

    #[config(default = 256)]
    pub batch_size: usize,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(artifact_dir: &str, device: B::Device) {
    create_artifact_dir(artifact_dir);

    // Config
    let optimizer = AdamConfig::new();
    let config = ExpConfig::new(optimizer);
    let model = RegressionModelConfig::new().init(&device);
    B::seed(&device, config.seed);

    // Define train/valid datasets and dataloaders
    let train_dataset = HousingDataset::train();
    let valid_dataset = HousingDataset::validation();

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Valid Dataset Size: {}", valid_dataset.len());

    let batcher_train = HousingBatcher::<B>::new(device.clone());

    let batcher_test = HousingBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    // Model
    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let result = training.launch(Learner::new(model, config.optimizer.init(), 1e-3));

    config
        .save(format!("{artifact_dir}/config.json").as_str())
        .unwrap();

    result
        .model
        .save_file(
            format!("{artifact_dir}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}
