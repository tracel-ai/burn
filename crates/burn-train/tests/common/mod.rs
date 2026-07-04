use std::sync::Arc;

use burn_core as burn;

use burn_core::{
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    module::{Module, Param},
    tensor::{Device, Distribution, Tensor},
};
use burn_optim::{ModuleOptimizer, SgdConfig, lr_scheduler::constant::ConstantLr};
use burn_train::{
    InferenceStep, Learner, LearningComponentsMarker, RegressionOutput, TrainOutput, TrainStep,
};

// Minimal toy model
#[derive(Module, Debug)]
pub struct ToyModel {
    pub weight: Param<Tensor<2>>,
}

impl ToyModel {
    pub fn new(device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::random([1, 2], Distribution::Default, device)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DummyBatch {
    target: Tensor<2>,
}

pub struct DummyBatcher;

impl Batcher<(), DummyBatch> for DummyBatcher {
    fn batch(&self, _items: Vec<()>, device: &Device) -> DummyBatch {
        DummyBatch {
            target: Tensor::zeros([1, 2], device),
        }
    }
}

impl TrainStep for ToyModel {
    type Input = DummyBatch;
    type Output = RegressionOutput;

    fn step(&self, item: DummyBatch) -> TrainOutput<RegressionOutput> {
        let output = self.weight.val();
        let loss = output
            .clone()
            .sub(item.target.clone())
            .powi_scalar(2)
            .mean();
        let regression = RegressionOutput::new(loss.clone(), output, item.target);
        TrainOutput::new(self, loss.backward(), regression)
    }
}

impl InferenceStep for ToyModel {
    type Input = DummyBatch;
    type Output = RegressionOutput;

    fn step(&self, item: DummyBatch) -> RegressionOutput {
        let output = self.weight.val();
        let loss = output
            .clone()
            .sub(item.target.clone())
            .powi_scalar(2)
            .mean();
        RegressionOutput::new(loss, output, item.target)
    }
}

pub type ToyLearner = Learner<LearningComponentsMarker<ConstantLr, ToyModel>>;

pub fn make_dataloaders() -> (
    Arc<dyn burn_core::data::dataloader::DataLoader<DummyBatch>>,
    Arc<dyn burn_core::data::dataloader::DataLoader<DummyBatch>>,
) {
    let dl_train = DataLoaderBuilder::new(DummyBatcher)
        .batch_size(2)
        .build(InMemDataset::new(vec![(); 4]));
    let dl_valid = DataLoaderBuilder::new(DummyBatcher)
        .batch_size(2)
        .build(InMemDataset::new(vec![(); 4]));
    (dl_train, dl_valid)
}

pub fn make_learner(device: &Device) -> ToyLearner {
    let model = ToyModel::new(device);
    let optim: ModuleOptimizer = SgdConfig::new().init();
    let scheduler = ConstantLr::new(1e-3);
    Learner::new(model, optim, scheduler)
}
