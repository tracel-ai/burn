use std::{sync::Arc, time::Duration};

use crate::{
    data::{MnistBatcher, MnistMapper},
    model::Model,
};

use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{
            transform::{ComposedDataset, MapperDataset, PartialDataset, SamplerDataset},
            vision::{MnistDataset, MnistItem},
        },
    },
    lr_scheduler::{
        composed::ComposedLrSchedulerConfig, cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        EvaluatorBuilder, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LearningRateMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
        renderer::MetricsRenderer,
    },
};

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

#[derive(Config)]
pub struct MnistTrainingConfig {
    #[config(default = 50)]
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

    let dataset_train_original = Arc::new(MnistDataset::train());
    let dataset_train_plain = PartialDataset::new(dataset_train_original.clone(), 0, 50_000);
    let dataset_valid_plain = PartialDataset::new(dataset_train_original.clone(), 50_000, 60_000);

    let data_augmentation = |dataset: PartialDataset<Arc<MnistDataset>, MnistItem>,
                             num_hard: usize,
                             num_medium: usize,
                             num_easy: usize| {
        // We create a dataset with a good distribution of samples of different difficulty.
        ComposedDataset::new(vec![
            // A dataset with all data augmentation activated.
            //
            // Will be hard for the model to learn.
            SamplerDataset::with_replacement(
                MapperDataset::new(
                    dataset.clone(),
                    MnistMapper::default()
                        .translate()
                        .shear()
                        .scale()
                        .rotation(),
                ),
                num_hard,
            ),
            // A dataset with some data augmentation activated.
            //
            // Will be somewhat hard for the model to learn.
            SamplerDataset::with_replacement(
                MapperDataset::new(
                    dataset.clone(),
                    MnistMapper::default().translate().rotation(),
                ),
                num_medium,
            ),
            // A dataset with no data augmentation activated.
            //
            // Will be easy for the model to learn.
            SamplerDataset::with_replacement(
                MapperDataset::new(dataset.clone(), MnistMapper::default()),
                num_easy,
            ),
        ])
    };

    let dataset_train = data_augmentation(dataset_train_plain, 25_000, 25_000, 10_000);
    let dataset_valid = data_augmentation(dataset_valid_plain, 2000, 2000, 1000);

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
        .consine(CosineAnnealingLrSchedulerConfig::new(1.0, 1000))
        // Warmup
        .linear(LinearLrSchedulerConfig::new(1e-8, 1.0, 1000))
        .linear(LinearLrSchedulerConfig::new(1e-2, 1e-6, 5000));

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metrics((
            AccuracyMetric::new(),
            LossMetric::new(),
            CpuUse::new(),
            CpuMemory::new(),
            CpuTemperature::new(),
            LearningRateMetric::new(),
        ))
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 20 },
        ))
        .num_epochs(config.num_epochs)
        .summary()
        .learning_strategy(burn::train::LearningStrategy::SingleDevice(device))
        .build(model, config.optimizer.init(), lr_scheduler.init().unwrap());

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

    let dataset_test_plain = Arc::new(MnistDataset::test());
    let evaluate = |name: &str,
                    mapper: MnistMapper,
                    model: Model<B::InnerBackend>,
                    renderer: Box<dyn MetricsRenderer>| {
        let batcher = MnistBatcher::default();
        let dataset_test = MapperDataset::new(dataset_test_plain.clone(), mapper);
        let dataloader_test = DataLoaderBuilder::new(batcher)
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .build(dataset_test);

        let evaluator = EvaluatorBuilder::new(ARTIFACT_DIR)
            .renderer(renderer)
            .metrics((AccuracyMetric::new(), LossMetric::new()))
            .build(model);

        evaluator.eval(name, dataloader_test)
    };

    let renderer = result.renderer;
    let renderer = evaluate(
        "Plain",
        MnistMapper::default(),
        result.model.clone(),
        renderer,
    );
    let renderer = evaluate(
        "Medium",
        MnistMapper::default().translate().rotation(),
        result.model.clone(),
        renderer,
    );

    let mut renderer = evaluate(
        "Hard",
        MnistMapper::default()
            .translate()
            .rotation()
            .shear()
            .scale(),
        result.model.clone(),
        renderer,
    );

    renderer.manual_close();
    core::mem::drop(renderer);

    // Making sure the Terminal is resetted.
    std::thread::sleep(Duration::from_secs(1));
    if let Some(summary) = result.summary {
        log::info!("{}", summary);
        println!("{}", summary);
    }
}
