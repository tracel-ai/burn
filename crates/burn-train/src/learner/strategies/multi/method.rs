use crate::{
    LearnerComponents, LearningMethod, TrainLoader, ValidLoader, components::LearnerComponentTypes,
    learner::strategies::single::epoch::SingleDeviceValidEpoch,
    multi::epoch::MultiDeviceTrainEpoch,
};
use burn_core::{data::dataloader::split::split_dataloader, module::Module, prelude::Backend};
use std::marker::PhantomData;

pub struct MultiDeviceLearningStrategy<LC: LearnerComponentTypes> {
    devices: Vec<<LC::Backend as Backend>::Device>,
    _p: PhantomData<LC>,
}
impl<LC: LearnerComponentTypes> MultiDeviceLearningStrategy<LC> {
    pub fn new(devices: Vec<<LC::Backend as Backend>::Device>) -> Self {
        Self {
            devices,
            _p: PhantomData,
        }
    }
}

impl<LC: LearnerComponentTypes> LearningMethod<LC> for MultiDeviceLearningStrategy<LC> {
    type PreparedDataloaders = (Vec<TrainLoader<LC>>, ValidLoader<LC>);

    type PreparedModel = LC::Model;

    fn prepare_dataloaders(
        &self,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
    ) -> Self::PreparedDataloaders {
        // `MultiDevicesTrainStep` has one worker per device, so we use a fixed device strategy
        // for each (worker) data loader. This matches the expected device on the worker, so we
        // don't have to move the data between devices.
        let train = split_dataloader(dataloader_train, &self.devices);
        let main_device = self.devices.first().unwrap();
        let valid = dataloader_valid.to_device(main_device);

        (train, valid)
    }

    fn prepare_model(&self, model: LC::Model) -> Self::PreparedModel {
        let main_device = self.devices.first().unwrap();
        model.fork(main_device)
    }

    fn learn(
        &self,
        mut model: LC::Model,
        (dataloader_train, dataloader_valid): Self::PreparedDataloaders,
        starting_epoch: usize,
        mut components: LearnerComponents<LC>,
    ) -> LC::Model {
        let mut epoch_train = MultiDeviceTrainEpoch::<LC>::new(
            dataloader_train,
            starting_epoch,
            components.num_epochs,
            components.grad_accumulation,
        );

        for epoch in starting_epoch..components.num_epochs + 1 {
            (model, components.optim) = epoch_train.run(
                model,
                components.optim,
                &mut components.lr_scheduler,
                components.event_processor,
                self.devices.to_vec(),
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
