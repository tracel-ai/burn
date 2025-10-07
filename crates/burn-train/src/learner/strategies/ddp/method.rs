use core::panic;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use burn_collective::CollectiveConfig;
use burn_core::prelude::Backend;

use crate::learner::strategies::ddp::DdpWorker;
use crate::{LearnerComponents, LearningMethod, TrainLoader, ValidLoader};
use burn_core::data::dataloader::split::split_dataloader;

use crate::components::LearnerComponentTypes;

pub struct DdpLearningStrategy<LC: LearnerComponentTypes> {
    devices: Vec<<LC::Backend as Backend>::Device>,
    config: CollectiveConfig,
    _p: PhantomData<LC>,
}
impl<LC: LearnerComponentTypes> DdpLearningStrategy<LC> {
    pub fn new(devices: Vec<<LC::Backend as Backend>::Device>, config: CollectiveConfig) -> Self {
        let config = config.with_num_devices(devices.len());
        Self {
            devices,
            config,
            _p: PhantomData,
        }
    }
}

impl<LC: LearnerComponentTypes + Send + 'static> LearningMethod<LC> for DdpLearningStrategy<LC> {
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
        model
    }

    fn learn(
        &self,
        model: Self::PreparedModel,
        dataloaders: Self::PreparedDataloaders,
        starting_epoch: usize,
        components: LearnerComponents<LC>,
    ) -> (LC::Model, LC::EventProcessor) {
        let (mut dataloaders_train, dataloader_valid) = dataloaders;
        let model: LC::Model = model;

        // The reference model is always on the first device provided.
        let main_device = self.devices[0].clone();
        let peer_count = self.devices.len();
        let event_processor = Arc::new(Mutex::new(components.event_processor));

        // Start worker for main device
        // First training dataloader corresponds to main device
        let main_handle = DdpWorker::<LC>::start(
            0.into(),
            main_device,
            model.clone(),
            components.optim.clone(),
            components.early_stopping.clone(),
            event_processor.clone(),
            components.event_store.clone(),
            components.checkpointer,
            components.lr_scheduler.clone(),
            components.interrupter.clone(),
            dataloaders_train.remove(0),
            Some(dataloader_valid),
            self.config.clone(),
            starting_epoch,
            components.num_epochs,
            components.grad_accumulation,
            peer_count,
            true,
        );

        // Spawn other workers for the other devices, starting with peer id 1
        let mut peer_id = 1;
        let mut secondary_workers = vec![];
        for device in &self.devices[1..] {
            let handle = DdpWorker::<LC>::start(
                peer_id.into(),
                device.clone(),
                model.clone(),
                components.optim.clone(),
                components.early_stopping.clone(),
                event_processor.clone(),
                components.event_store.clone(),
                None,
                components.lr_scheduler.clone(),
                components.interrupter.clone(),
                dataloaders_train.remove(0),
                None,
                self.config.clone(),
                starting_epoch,
                components.num_epochs,
                components.grad_accumulation,
                peer_count,
                false,
            );

            peer_id += 1;

            secondary_workers.push(handle);
        }

        // Wait for all devices to finish
        for worker in secondary_workers {
            worker
                .join()
                .expect("Distributed data parallel worker failed");
        }
        // Main worker had the event processor
        let model = main_handle
            .join()
            .expect("Distributed data parallel main worker failed");

        let Ok(event_processor) = Arc::try_unwrap(event_processor) else {
            panic!("Event processor still held!");
        };
        let Ok(event_processor) = event_processor.into_inner() else {
            panic!("Event processor lock poisoned");
        };
        (model, event_processor)
    }
}
