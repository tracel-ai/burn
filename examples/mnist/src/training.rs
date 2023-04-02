use std::sync::Arc;

use crate::data::MNISTBatcher;
use crate::model::Model;

use burn::module::Module;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{Adam, AdamConfig};
use burn::record::{DefaultRecordSettings, NoStdTrainingRecordSettings, Record};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::source::huggingface::MNISTDataset},
    tensor::backend::ADBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 4)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

pub fn run<B: ADBackend>(device: B::Device) {
    // Config
    let config_optimizer =
        AdamConfig::new(1e-4).with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config = MnistTrainingConfig::new(config_optimizer);
    B::seed(config.seed);

    // Data
    let batcher_train = Arc::new(MNISTBatcher::<B>::new(device.clone()));
    let batcher_valid = Arc::new(MNISTBatcher::<B::InnerBackend>::new(device.clone()));
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Arc::new(MNISTDataset::train()));
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Arc::new(MNISTDataset::test()));

    // Model
    let optim = Adam::new(&config.optimizer);
    let model = Model::new();

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_plot(AccuracyMetric::new())
        .metric_valid_plot(AccuracyMetric::new())
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .with_file_checkpointer::<DefaultRecordSettings>(2)
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, optim);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .into_record()
        .record::<NoStdTrainingRecordSettings>(format!("{ARTIFACT_DIR}/model").into())
        .expect("Failed to save trained model");
}
