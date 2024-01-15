use burn::module::Module;
use burn::optim::SgdConfig;
use burn::record::{CompactRecorder, NoStdTrainingRecorder};
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{CpuMemory, CpuTemperature, CpuUse};
use burn::train::{MetricEarlyStoppingStrategy, StoppingCondition};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::LossMetric,
        LearnerBuilder,
    },
};
use burn::data::dataset::Dataset;
use burn::tensor::backend::Backend;
use crate::dataset::{DiabetesBatcher, DiabetesDataset};
use crate::model::LinearModel;

static ARTIFACT_DIR: &str = "/tmp/burn-example-regression";

#[derive(Config)]
pub struct RegressionConfig {
    #[config(default = 5)]
    pub num_epochs: usize,

    #[config(default = 1)]
    pub batch_size: usize,

    #[config(default = 1)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: SgdConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    // Config
    let config_optimizer = SgdConfig::new();
    let config = RegressionConfig::new(config_optimizer);
    B::seed(config.seed);

    // Data
    let train_dataset = DiabetesDataset::train();

    println!(" Train {:?}", train_dataset.get(0).unwrap());

    let batcher_train = DiabetesBatcher::<B>::new(device.clone());

    let batcher_test = DiabetesBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DiabetesDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DiabetesDataset::test());

    // println!(" Test {}" ,dataloader_test.len());
    // println!(" Train {}" ,dataloader_train.len());

    // for (iteration, batch) in dataloader_train.iter().enumerate() {
    //     println!("Iteration: {}", iteration);
    //     println!("Batch: {:?} : {:?} ", batch.inputs.shape(), batch.targets.shape());
    //     break;
    // }

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        // .metric_train_numeric(CpuUse::new())
        // .metric_valid_numeric(CpuUse::new())

        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(LinearModel::new(&device), config.optimizer.init(), 1e-4);

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
