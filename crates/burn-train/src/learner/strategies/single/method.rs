use crate::{
    LearnerComponents, LearningMethod, SupervisedLearningTypes, TrainLoader, TrainStep,
    ValidLoader, ValidStep,
    components::{LearnerComponentTypes, LearningData},
    learner::strategies::single::epoch::{SingleDeviceTrainEpoch, SingleDeviceValidEpoch},
};
use burn_core::{module::Module, tensor::Device};
use std::{marker::PhantomData, sync::Arc};

/// Simplest learning strategy possible, with only a single devices doing both the training and
/// validation.
pub struct SingleDeviceLearningStrategy<LC: LearnerComponentTypes> {
    device: Device<LC::Backend>,
    dataloader_train: TrainLoader<LC>,
    dataloader_valid: ValidLoader<LC>,
    _p: PhantomData<LC>,
}
impl<LC: LearnerComponentTypes> SingleDeviceLearningStrategy<LC> {
    pub fn new(device: Device<LC::Backend>) -> Self {
        todo!()
        // Self {
        //     device,
        //     _p: PhantomData,
        // }
    }
}

pub type CustomSingleDeviceLearningStrategy<LC> = Arc<dyn LearningMethod<LC>>;

impl<SL: SupervisedLearningTypes> LearningMethod<SL> for SingleDeviceLearningStrategy<SL::LC> {
    fn learn(
        &self,
        model: SL::Model,
        starting_epoch: usize,
        mut components: LearnerComponents<SL::LC>,
    ) -> (SL::Model, <SL::LC as LearnerComponentTypes>::EventProcessor) {
        let mut model = model.fork(&self.device);
        let dataloader_train = self.dataloader_train.to_device(&self.device);
        let dataloader_valid = self.dataloader_valid.to_device(&self.device);

        let mut epoch_train = SingleDeviceTrainEpoch::new(
            dataloader_train,
            starting_epoch,
            components.num_epochs,
            components.grad_accumulation,
        );

        for epoch in starting_epoch..components.num_epochs + 1 {
            (model, components.optim) = epoch_train.run::<SL::LC>(
                model,
                components.optim,
                &mut components.lr_scheduler,
                &mut components.event_processor,
                &components.interrupter,
            );

            if components.interrupter.should_stop() {
                break;
            }

            let epoch_valid = SingleDeviceValidEpoch::<SL::LC>::new(
                dataloader_valid.clone(),
                epoch,
                components.num_epochs,
            );
            epoch_valid.run(
                &model,
                &mut components.event_processor,
                &components.interrupter,
            );

            if let Some(checkpointer) = &mut components.checkpointer {
                checkpointer.checkpoint(
                    &model,
                    &components.optim,
                    &components.lr_scheduler,
                    epoch,
                    &components.event_store,
                );
            }

            if let Some(early_stopping) = &mut components.early_stopping
                && early_stopping.should_stop(epoch, &components.event_store)
            {
                break;
            }
        }

        (model, components.event_processor)
    }
}
