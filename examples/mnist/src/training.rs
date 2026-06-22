use std::sync::Arc;

use crate::{
    data::{MnistBatcher, MnistItemPrepared, MnistMapper, Transform},
    file_progress::{FileEvaluationProgressLogger, FileTrainingProgressLogger},
    model::Model,
};

use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{
            Dataset,
            transform::{ComposedDataset, MapperDataset, PartialDataset, SamplerDataset},
            vision::{MnistDataset, MnistItem},
        },
    },
    lr_scheduler::{
        composed::ComposedLrSchedulerConfig, cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    prelude::*,
    train::{
        EvaluatorBuilder, Learner, MetricEarlyStoppingStrategy, StoppingCondition,
        metric::{
            AccuracyMetric, LearningRateMetric, LossMetric,
            store::{Aggregate, Direction, Split},
        },
    },
};
use burn::{optim::AdamWConfig, train::SupervisedTraining};

static ARTIFACT_DIR: &str = "/tmp/burn-example-mnist";

#[derive(Config, Debug)]
pub struct MnistTrainingConfig {
    #[config(default = 5)]
    pub num_epochs: usize,

    #[config(default = 256)]
    pub batch_size: usize,

    #[config(default = 8)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamWConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn run(device: Device) {
    create_artifact_dir(ARTIFACT_DIR);
    // Config
    let config_optimizer = AdamWConfig::new()
        .with_cautious_weight_decay(true)
        .with_weight_decay(5e-5);

    let config = MnistTrainingConfig::new(config_optimizer);

    device.seed(config.seed);
    let autodiff_device = device.clone().autodiff();

    let model = Model::new(&autodiff_device);

    let dataset_train_original = Arc::new(MnistDataset::train());
    let dataset_train_plain = PartialDataset::new(dataset_train_original.clone(), 0, 55_000);
    let dataset_valid_plain = PartialDataset::new(dataset_train_original.clone(), 55_000, 60_000);

    let ident_trains = generate_idents(Some(10000));
    let ident_valid = generate_idents(None);
    let dataset_train = DatasetIdent::compose(ident_trains, dataset_train_plain);
    let dataset_valid = DatasetIdent::compose(ident_valid, dataset_valid_plain);

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

    let training = SupervisedTraining::new(ARTIFACT_DIR, dataloader_train, dataloader_valid)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .metric_train_numeric(LearningRateMetric::new())
        .with_default_checkpointers()
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 5 },
        ))
        .with_progress_logger(
            FileTrainingProgressLogger::new(format!("{ARTIFACT_DIR}/training_progress.log"))
                .expect("Failed to create training progress log"),
        )
        .num_epochs(config.num_epochs)
        .summary();

    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        lr_scheduler.init().unwrap(),
    ));

    let dataset_test_plain = Arc::new(MnistDataset::test());

    let splits: Vec<_> = generate_idents(None)
        .into_iter()
        .map(|(ident, _)| {
            let name = ident.to_string();
            let dataset_test = DatasetIdent::prepare(ident, dataset_test_plain.clone());
            let dataloader = DataLoaderBuilder::new(MnistBatcher::default())
                .batch_size(config.batch_size)
                .num_workers(2)
                .build(dataset_test);
            (name, dataloader)
        })
        .collect();

    let mut renderer = EvaluatorBuilder::new(ARTIFACT_DIR)
        .renderer(result.renderer)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_progress_logger(
            FileEvaluationProgressLogger::new(format!("{ARTIFACT_DIR}/evaluation_progress.log"))
                .expect("Failed to create evaluation progress log"),
        )
        .summary()
        .build(result.model.clone())
        .eval_all(splits);

    result
        .model
        .into_record()
        .save(format!("{ARTIFACT_DIR}/model"))
        .expect("Failed to save trained model");

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    renderer.manual_close();
}

enum DatasetIdent {
    Plain,
    Transformed(Vec<Transform>),
    All,
}

impl core::fmt::Display for DatasetIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetIdent::Plain => f.write_str("Plain")?,
            DatasetIdent::Transformed(items) => {
                for i in 0..items.len() {
                    f.write_fmt(format_args!("{}", items[i]))?;
                    if i < items.len() - 1 {
                        f.write_str(" ")?;
                    }
                }
            }
            DatasetIdent::All => f.write_str("All")?,
        };

        Ok(())
    }
}

impl DatasetIdent {
    pub fn many(transforms: Vec<Transform>) -> Self {
        Self::Transformed(transforms)
    }

    pub fn prepare(self, dataset: impl Dataset<MnistItem>) -> impl Dataset<MnistItemPrepared> {
        let items = match self {
            DatasetIdent::Plain => Vec::new(),
            DatasetIdent::All => {
                vec![
                    Transform::Translate,
                    Transform::Shear,
                    Transform::Scale,
                    Transform::Rotation,
                ]
            }
            DatasetIdent::Transformed(items) => items.clone(),
        };
        MapperDataset::new(dataset, MnistMapper::default().transform(&items))
    }

    pub fn compose(
        idents: Vec<(Self, Option<usize>)>,
        dataset: PartialDataset<Arc<MnistDataset>, MnistItem>,
    ) -> impl Dataset<MnistItemPrepared> {
        let datasets = idents
            .into_iter()
            .map(|(ident, size)| match size {
                Some(size) => {
                    SamplerDataset::with_replacement(ident.prepare(dataset.clone()), size)
                }
                None => {
                    let dataset = ident.prepare(dataset.clone());
                    let size = dataset.len();
                    SamplerDataset::without_replacement(dataset, size)
                }
            })
            .collect();
        ComposedDataset::new(datasets)
    }
}

fn generate_idents(num_samples_base: Option<usize>) -> Vec<(DatasetIdent, Option<usize>)> {
    let mut current = Vec::new();
    let mut idents = Vec::new();

    for shear in [None, Some(Transform::Shear)] {
        for scale in [None, Some(Transform::Scale)] {
            for rotation in [None, Some(Transform::Rotation)] {
                for translate in [None, Some(Transform::Translate)] {
                    if let Some(tr) = shear {
                        current.push(tr);
                    }
                    if let Some(tr) = scale {
                        current.push(tr);
                    }
                    if let Some(tr) = rotation {
                        current.push(tr);
                    }
                    if let Some(tr) = translate {
                        current.push(tr);
                    }

                    let num_samples = num_samples_base.map(|val| val * current.len());

                    if current.len() == 4 {
                        idents.push((DatasetIdent::All, num_samples));
                    } else if current.is_empty() {
                        idents.push((DatasetIdent::Plain, num_samples));
                    } else {
                        idents.push((DatasetIdent::many(current.clone()), num_samples));
                    }

                    current.clear();
                }
            }
        }
    }

    idents
}
