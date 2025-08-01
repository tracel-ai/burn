use burn_core::tensor::backend::AutodiffBackend;

use crate::{
    Learner, LearnerDataLoaders, TrainLoader, ValidLoader, components::LearnerComponents,
    learn_multi_device, learn_single_device, prepare_dataloaders_multi_device,
    prepare_dataloaders_single_device, prepare_model_multi_device, prepare_model_single_device,
};

/// How should the learner run the learning for the model
#[derive(Clone)]
pub enum LearningStrategy<B: AutodiffBackend> {
    /// Training on one device
    SingleDevice(B::Device),
    /// Legacy implementation of local multi-device training
    MultiDeviceNaive(Vec<B::Device>),
    // DistributedDataParallel...,
}

impl<B: AutodiffBackend> Default for LearningStrategy<B> {
    fn default() -> Self {
        Self::SingleDevice(Default::default())
    }
}

pub(crate) trait LearningStrategyExt<LC: LearnerComponents> {
    fn prepare_dataloaders(
        &self,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
    ) -> LearnerDataLoaders<LC>;

    fn prepare_model(&self, learner: Learner<LC>) -> Learner<LC>;

    fn learn(
        &self,
        learner: Learner<LC>,
        dataloaders: LearnerDataLoaders<LC>,
        starting_epoch: usize,
    ) -> Learner<LC>;
}

impl<LC: LearnerComponents> LearningStrategyExt<LC> for LearningStrategy<LC::Backend> {
    fn prepare_dataloaders(
        &self,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
    ) -> LearnerDataLoaders<LC> {
        match self {
            LearningStrategy::SingleDevice(device) => {
                prepare_dataloaders_single_device::<LC>(device, dataloader_train, dataloader_valid)
            }
            LearningStrategy::MultiDeviceNaive(devices) => {
                prepare_dataloaders_multi_device::<LC>(devices, dataloader_train, dataloader_valid)
            }
        }
    }

    fn prepare_model(&self, learner: Learner<LC>) -> Learner<LC> {
        match self {
            LearningStrategy::SingleDevice(device) => {
                prepare_model_single_device::<LC>(device, learner)
            }
            LearningStrategy::MultiDeviceNaive(devices) => {
                prepare_model_multi_device::<LC>(devices, learner)
            }
        }
    }

    fn learn(
        &self,
        learner: Learner<LC>,
        dataloaders: LearnerDataLoaders<LC>,
        starting_epoch: usize,
    ) -> Learner<LC> {
        match self {
            LearningStrategy::SingleDevice(_device) => {
                learn_single_device::<LC>(learner, dataloaders, starting_epoch)
            }
            LearningStrategy::MultiDeviceNaive(devices) => {
                learn_multi_device::<LC>(devices, learner, dataloaders, starting_epoch)
            }
        }
    }
}
