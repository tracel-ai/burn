use crate::{
    LearnerComponents, LearningMethod, TrainLoader, ValidLoader,
    components::LearnerComponentTypes,
    learner::strategies::single::epoch::{SingleDeviceTrainEpoch, SingleDeviceValidEpoch},
};
use burn_core::{module::Module, tensor::Device};
use std::{marker::PhantomData, sync::Arc};

/// Simplest learning strategy possible, with only a single devices doing both the training and
/// validation.
pub struct SingleDeviceLearningStrategy<LC: LearnerComponentTypes> {
    device: Device<LC::Backend>,
    _p: PhantomData<LC>,
}
impl<LC: LearnerComponentTypes> SingleDeviceLearningStrategy<LC> {
    pub fn new(device: Device<LC::Backend>) -> Self {
        Self {
            device,
            _p: PhantomData,
        }
    }
}

pub type CustomSingleDeviceLearningStrategy<LC> = Arc<
    dyn LearningMethod<
            LC,
            PreparedDataloaders = (TrainLoader<LC>, ValidLoader<LC>),
            PreparedModel = <LC as LearnerComponentTypes>::Model,
        >,
>;

impl<LC: LearnerComponentTypes> LearningMethod<LC> for SingleDeviceLearningStrategy<LC> {
    type PreparedDataloaders = (TrainLoader<LC>, ValidLoader<LC>);

    type PreparedModel = LC::Model;

    fn prepare_dataloaders(
        &self,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
    ) -> Self::PreparedDataloaders {
        // The reference model is always on the first device provided.
        let train = dataloader_train.to_device(&self.device);
        let valid = dataloader_valid.to_device(&self.device);

        (train, valid)
    }

    fn prepare_model(&self, model: LC::Model) -> Self::PreparedModel {
        model.fork(&self.device)
    }

    fn learn(
        &self,
        mut model: LC::Model,
        (dataloader_train, dataloader_valid): Self::PreparedDataloaders,
        starting_epoch: usize,
        mut components: LearnerComponents<LC>,
    ) -> (LC::Model, LC::EventProcessor) {
        let mut epoch_train = SingleDeviceTrainEpoch::new(
            dataloader_train,
            starting_epoch,
            components.num_epochs,
            components.grad_accumulation,
        );

        for epoch in starting_epoch..components.num_epochs + 1 {
            (model, components.optim) = epoch_train.run::<LC>(
                model,
                components.optim,
                &mut components.lr_scheduler,
                &mut components.event_processor,
                &components.interrupter,
            );

            if components.interrupter.should_stop() {
                break;
            }

            let epoch_valid = SingleDeviceValidEpoch::<LC>::new(
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
