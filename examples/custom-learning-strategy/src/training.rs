use burn::train::{EventProcessorTraining, Training};

use std::{marker::PhantomData, sync::Arc};

use crate::model::ModelConfig;
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{transform::PartialDataset, vision::MnistDataset},
    },
    lr_scheduler::{
        LrScheduler, composed::ComposedLrSchedulerConfig, cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    module::AutodiffModule,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{Device, backend::AutodiffBackend},
    train::{
        LearnerBuilder, LearnerComponentTypes, LearnerEvent, LearnerItem, LearningMethod,
        MetricEarlyStoppingStrategy, StoppingCondition, TrainLoader, TrainStep, ValidLoader,
        ValidStep,
        metric::{
            AccuracyMetric, LearningRateMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use guide::data::MnistBatcher;

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

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

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    println!("Run method of custom-learning-strat");

    create_artifact_dir(ARTIFACT_DIR);
    // Config
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);

    B::seed(&device, config.seed);

    let model = config.model.init::<B>(&device);

    let dataset_train_original = Arc::new(MnistDataset::train());
    let dataset_train = PartialDataset::new(dataset_train_original.clone(), 0, 55_000);
    let dataset_valid = PartialDataset::new(dataset_train_original.clone(), 55_000, 60_000);

    let dataloader_train = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_train);
    let dataloader_valid = DataLoaderBuilder::new(MnistBatcher::default())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(dataset_valid);
    let lr_scheduler = ComposedLrSchedulerConfig::new()
        .cosine(CosineAnnealingLrSchedulerConfig::new(1.0, 2000))
        // Warmup
        .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, 2000))
        .linear(LinearLrSchedulerConfig::new(1e-2, 1e-6, 10000));
    let early_stopping = MetricEarlyStoppingStrategy::new(
        &LossMetric::<B>::new(),
        Aggregate::Mean,
        Direction::Lowest,
        Split::Valid,
        StoppingCondition::NoImprovementSince { n_epochs: 5 },
    );

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(early_stopping)
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), lr_scheduler.init().unwrap());

    learner.study(Knowledge::new(device, dataloader_train, dataloader_valid));
}

struct Experience<LC: LearnerComponentTypes> {
    device: Device<LC::Backend>,
    dataloader_train: TrdinLoader<LC>,
    dataloader_valid: ValidLoader<LC>,
    _p: PhantomData<LC>,
}

impl<LC: LearnerComponentTypes> Experience<LC> {
    pub fn new(
        device: Device<LC::Backend>,
        dataloader_train: TrdinLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
    ) -> Self {
        Self {
            device,
            dataloader_train,
            dataloader_valid,
            _p: PhantomData,
        }
    }
}

impl<LC: LearnerComponentTypes> LearningMethod<LC> for Experience<LC> {
    type PreparedDataloaders = (TrainLoader<LC>, ValidLoader<LC>);

    type PreparedModel = <LC as LearnerComponentTypes>::Model;

    fn prepare_dataloaders(&self) -> Self::PreparedDataloaders {
        // The reference model is always on the first device provided.
        let train = self.dataloader_train.to_device(&self.device);
        let valid = self.dataloader_valid.to_device(&self.device);

        (train, valid)
    }

    fn prepare_model(&self, model: LC::Model) -> Self::PreparedModel {
        model.fork(&self.device)
    }

    fn learn(
        &self,
        mut model: Self::PreparedModel,
        (dataloader_train, dataloader_valid): Self::PreparedDataloaders,
        starting_epoch: usize,
        components: burn::train::LearnerComponents<LC>,
    ) -> (LC::Model, LC::EventProcessor) {
        let mut scheduler = components.lr_scheduler;
        let mut optim = components.optim;
        let mut processor = components.event_processor;

        for epoch in starting_epoch..components.num_epochs + 1 {
            // Iterate over our training and validation loop for X epochs.
            log::info!("Executing training step for epoch {}", epoch,);

            // Single device / dataloader
            let mut iterator = dataloader_train.iter();
            let mut iteration = 0;

            while let Some(item) = iterator.next() {
                iteration += 1;
                let lr = scheduler.step();
                log::info!("Iteration {iteration}");

                let progress = iterator.progress();
                let item = model.step(item);
                model = model.optimize(&mut optim, lr, item.grads);

                let item = LearnerItem::new(
                    item.item,
                    progress,
                    epoch,
                    components.num_epochs,
                    iteration,
                    Some(lr),
                );

                processor.process_train(LearnerEvent::ProcessedItem(item));
            }
            processor.process_train(LearnerEvent::EndEpoch(epoch));

            log::info!("Executing validation step for epoch {}", epoch);
            let model_valid = model.valid();

            let mut iterator = dataloader_valid.iter();
            let mut iteration = 0;

            while let Some(item) = iterator.next() {
                let progress = iterator.progress();
                iteration += 1;

                let item = model_valid.step(item);
                let item = LearnerItem::new(
                    item,
                    progress,
                    epoch,
                    components.num_epochs,
                    iteration,
                    None,
                );

                processor.process_valid(LearnerEvent::ProcessedItem(item));
            }
            processor.process_valid(LearnerEvent::EndEpoch(epoch));
        }

        (model, processor)
    }
}
