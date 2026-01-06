use crate::model::ModelConfig;
use burn::record::NoStdTrainingRecorder;
use burn::train::{
    EventProcessorTraining, Learner, LearnerBackend, LearnerModel, ParadigmComponentsTypes,
    SupervisedLearningComponentsTypes, SupervisedLearningStrategy, SupervisedTraining, TrainLoader,
    TrainingComponents, ValidLoader,
};
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{transform::PartialDataset, vision::MnistDataset},
    },
    lr_scheduler::{
        composed::ComposedLrSchedulerConfig, cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    module::AutodiffModule,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{Device, backend::AutodiffBackend},
    train::{
        LearnerEvent, LearnerItem, MetricEarlyStoppingStrategy, StoppingCondition, ValidStep,
        metric::{
            AccuracyMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use guide::data::MnistBatcher;
use std::{marker::PhantomData, sync::Arc};

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

#[derive(Config, Debug)]
pub struct MnistTrainingConfig {
    #[config(default = 5)]
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
    create_artifact_dir(ARTIFACT_DIR);
    // Config
    let config_model = ModelConfig::new(10, 1024);
    let config_optimizer = AdamConfig::new();
    let config = MnistTrainingConfig::new(config_model, config_optimizer);

    B::seed(&device, config.seed);

    let model = config.model.init::<B>(&device);

    let dataset_train_original = Arc::new(MnistDataset::train());
    let dataset_train = PartialDataset::new(dataset_train_original.clone(), 0, 15_000);
    let dataset_valid = PartialDataset::new(dataset_train_original.clone(), 15_000, 17_000);

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

    let training = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_valid)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(early_stopping)
        .num_epochs(config.num_epochs)
        .summary()
        .with_training_strategy(burn::train::TrainingStrategy::Custom(Arc::new(
            MyCustomLearningStrategy::new(device),
        )));

    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        lr_scheduler.init().unwrap(),
    ));

    result
        .model
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}

struct MyCustomLearningStrategy<SC: SupervisedLearningComponentsTypes> {
    device: Device<LearnerBackend<SC::LC>>,
    _p: PhantomData<SC>,
}

impl<SC: SupervisedLearningComponentsTypes> MyCustomLearningStrategy<SC> {
    pub fn new(device: Device<LearnerBackend<SC::LC>>) -> Self {
        Self {
            device,
            _p: PhantomData,
        }
    }
}

impl<SC: SupervisedLearningComponentsTypes> SupervisedLearningStrategy<SC>
    for MyCustomLearningStrategy<SC>
{
    fn fit(
        &self,
        training_components: TrainingComponents<SC>,
        mut learner: Learner<SC::LC>,
        dataloader_train: TrainLoader<SC::LC>,
        dataloader_valid: ValidLoader<SC::LC>,
        starting_epoch: usize,
    ) -> (
        LearnerModel<SC::LC>,
        <SC::PC as ParadigmComponentsTypes>::EventProcessor,
    ) {
        let dataloader_train = dataloader_train.to_device(&self.device);
        let dataloader_valid = dataloader_valid.to_device(&self.device);
        learner.fork(&self.device);
        let mut event_processor = training_components.event_processor;
        let mut checkpointer = training_components.checkpointer;
        let interrupter = training_components.interrupter;
        let num_epochs = training_components.num_epochs;

        for epoch in starting_epoch..num_epochs + 1 {
            // Iterate over our training and validation loop for X epochs.
            log::info!("Executing training step for epoch {}", epoch,);

            // Single device / dataloader
            let mut iterator = dataloader_train.iter();
            let mut iteration = 0;

            while let Some(item) = iterator.next() {
                iteration += 1;
                learner.lr_step();
                log::info!("Iteration {iteration} of my custom learning strategy");

                let progress = iterator.progress();
                let item = learner.step(item);
                learner.optimize(item.grads);

                let item = LearnerItem::new(
                    item.item,
                    progress,
                    epoch,
                    num_epochs,
                    iteration,
                    Some(learner.lr_current()),
                );

                event_processor.process_train(LearnerEvent::ProcessedItem(item));

                if interrupter.should_stop() {
                    let reason = interrupter
                        .get_message()
                        .unwrap_or(String::from("Reason unknown"));
                    log::info!("Training interrupted: {reason}");
                    break;
                }
            }
            event_processor.process_train(LearnerEvent::EndEpoch(epoch));

            let model_valid = learner.model().valid();

            let mut iterator = dataloader_valid.iter();
            let mut iteration = 0;

            while let Some(item) = iterator.next() {
                let progress = iterator.progress();
                iteration += 1;

                let item = model_valid.step(item);
                let item = LearnerItem::new(item, progress, epoch, num_epochs, iteration, None);

                event_processor.process_valid(LearnerEvent::ProcessedItem(item));
            }
            event_processor.process_valid(LearnerEvent::EndEpoch(epoch));

            if let Some(checkpointer) = &mut checkpointer {
                checkpointer.checkpoint(&learner, epoch, &training_components.event_store);
            }
        }

        (learner.model(), event_processor)
    }
}
