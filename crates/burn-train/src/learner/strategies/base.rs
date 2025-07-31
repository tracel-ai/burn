use burn_core::{module::AutodiffModule, tensor::backend::AutodiffBackend};

use crate::{
    Learner, LearnerDataLoaders, TrainLoader, TrainStep, ValidLoader, ValidStep,
    components::LearnerComponents, learn_multi_device, learn_single_device,
    metric::processor::EventProcessor, prepare_dataloaders_multi_device,
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

pub(crate) trait LearningStrategyExt<B: AutodiffBackend> {
    fn prepare_dataloaders<LC, TI, VI>(
        &self,
        dataloader_train: TrainLoader<LC, TI>,
        dataloader_valid: ValidLoader<LC, VI>,
    ) -> LearnerDataLoaders<LC, TI, VI>
    where
        LC: LearnerComponents<Backend = B>;

    fn prepare_model<LC: LearnerComponents<Backend = B>>(
        &self,
        learner: Learner<LC>,
    ) -> Learner<LC>;

    fn learn<LC, TI, VI, TO, VO>(
        &self,
        learner: Learner<LC>,
        dataloaders: LearnerDataLoaders<LC, TI, VI>,
        starting_epoch: usize,
    ) -> Learner<LC>
    where
        TI: Send + 'static,
        TO: Send + 'static,
        LC: LearnerComponents<Backend = B>,
        LC::EventProcessor: EventProcessor<ItemValid = VO, ItemTrain = TO>,
        <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<VI, VO>,
        <LC as LearnerComponents>::Model: TrainStep<TI, TO>;
}

impl<B: AutodiffBackend> LearningStrategyExt<B> for LearningStrategy<B> {
    fn prepare_dataloaders<LC, TI, VI>(
        &self,
        dataloader_train: TrainLoader<LC, TI>,
        dataloader_valid: ValidLoader<LC, VI>,
    ) -> LearnerDataLoaders<LC, TI, VI>
    where
        LC: LearnerComponents<Backend = B>,
    {
        match self {
            LearningStrategy::SingleDevice(device) => {
                prepare_dataloaders_single_device::<LC, TI, VI>(
                    device,
                    dataloader_train,
                    dataloader_valid,
                )
            }
            LearningStrategy::MultiDeviceNaive(devices) => {
                prepare_dataloaders_multi_device::<LC, TI, VI>(
                    devices,
                    dataloader_train,
                    dataloader_valid,
                )
            }
        }
    }

    fn prepare_model<LC: LearnerComponents<Backend = B>>(
        &self,
        learner: Learner<LC>,
    ) -> Learner<LC> {
        match self {
            LearningStrategy::SingleDevice(device) => {
                prepare_model_single_device::<LC>(device, learner)
            }
            LearningStrategy::MultiDeviceNaive(devices) => {
                prepare_model_multi_device::<LC>(devices, learner)
            }
        }
    }

    fn learn<LC, TI, VI, TO, VO>(
        &self,
        learner: Learner<LC>,
        dataloaders: LearnerDataLoaders<LC, TI, VI>,
        starting_epoch: usize,
    ) -> Learner<LC>
    where
        TI: Send + 'static,
        TO: Send + 'static,
        LC: LearnerComponents<Backend = B>,
        LC::EventProcessor: EventProcessor<ItemValid = VO, ItemTrain = TO>,
        <LC::Model as AutodiffModule<LC::Backend>>::InnerModule: ValidStep<VI, VO>,
        <LC as LearnerComponents>::Model: TrainStep<TI, TO>,
    {
        match self {
            LearningStrategy::SingleDevice(_device) => {
                learn_single_device::<LC, TI, VI, TO, VO>(learner, dataloaders, starting_epoch)
            }
            LearningStrategy::MultiDeviceNaive(devices) => {
                learn_multi_device::<LC, TI, VI, TO, VO>(
                    devices,
                    learner,
                    dataloaders,
                    starting_epoch,
                )
            }
        }
    }
}
