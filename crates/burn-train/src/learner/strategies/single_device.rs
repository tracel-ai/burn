use std::marker::PhantomData;

use burn_core::{module::Module, prelude::Backend};

use crate::{
    LearnComponents, LearningMethod, SingleDeviceTrainEpoch, SingleDeviceValidEpoch, TrainLoader,
    ValidLoader, components::LearnerComponents,
};

pub struct SingleDeviceLearningStrategy<LC: LearnerComponents> {
    device: <LC::Backend as Backend>::Device,
    _p: PhantomData<LC>,
}
impl<LC: LearnerComponents> SingleDeviceLearningStrategy<LC> {
    pub fn new(device: <LC::Backend as Backend>::Device) -> Self {
        Self {
            device,
            _p: PhantomData,
        }
    }
}

impl<LC: LearnerComponents> LearningMethod<LC> for SingleDeviceLearningStrategy<LC> {
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
        model: Self::PreparedModel,
        dataloaders: Self::PreparedDataloaders,
        starting_epoch: usize,
        mut components: LearnComponents<LC>,
    ) -> LC::Model {
        let (dataloader_train, dataloader_valid) = dataloaders;
        let mut model: LC::Model = model;

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
                components.event_processor,
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
            epoch_valid.run(&model, components.event_processor, &components.interrupter);

            if let Some(checkpointer) = &mut components.checkpointer {
                checkpointer.checkpoint(
                    &model,
                    &components.optim,
                    &components.lr_scheduler,
                    epoch,
                    &components.event_store,
                );
            }

            if let Some(early_stopping) = &mut components.early_stopping {
                if early_stopping.should_stop(epoch, &components.event_store) {
                    break;
                }
            }
        }

        model
    }
}
