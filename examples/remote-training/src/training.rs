use std::sync::Arc;
use std::time::Instant;

use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{
            transform::{MapperDataset, PartialDataset},
            vision::MnistDataset,
        },
    },
    optim::AdamWConfig,
    train::{
        ExecutionStrategy, Learner, SupervisedTraining,
        metric::{AccuracyMetric, LossMetric},
    },
};
use mnist::{
    data::{MnistBatcher, MnistMapper},
    model::Model,
};

static ARTIFACT_DIR: &str = "/tmp/burn-remote-training";

const SEED: u64 = 42;
const TRAIN_ITEMS: usize = 55_000;
const VALID_ITEMS: usize = 5_000;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub num_epochs: usize,
    pub num_workers: usize,
}

/// Train the MNIST model under the given execution strategy and report wall-clock throughput.
pub fn run(strategy: ExecutionStrategy, config: TrainConfig) {
    let main_device = strategy.main_device().clone();
    main_device.seed(SEED);
    let model = Model::new(&main_device.autodiff());

    let dataset = Arc::new(MnistDataset::train());
    let dataset_train = MapperDataset::new(
        PartialDataset::new(dataset.clone(), 0, TRAIN_ITEMS),
        MnistMapper::default(),
    );
    let dataset_valid = MapperDataset::new(
        PartialDataset::new(dataset, TRAIN_ITEMS, TRAIN_ITEMS + VALID_ITEMS),
        MnistMapper::default(),
    );

    let dataloader_train = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(SEED)
        .num_workers(config.num_workers)
        .build(dataset_train);
    let dataloader_valid = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(SEED)
        .num_workers(config.num_workers)
        .build(dataset_valid);

    let training = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_valid)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_training_strategy(strategy.into())
        .num_epochs(config.num_epochs)
        .summary();

    let started = Instant::now();
    let result = training.launch(Learner::new(model, AdamWConfig::new().init(), 1.0e-3));
    let elapsed = started.elapsed();

    let items = TRAIN_ITEMS * config.num_epochs;
    println!(
        "trained on {items} items in {:.1}s ({:.0} items/s, including per-epoch validation)",
        elapsed.as_secs_f64(),
        items as f64 / elapsed.as_secs_f64(),
    );

    drop(result);
}
