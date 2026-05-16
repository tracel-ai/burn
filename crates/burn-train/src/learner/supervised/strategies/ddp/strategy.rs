use core::panic;
use std::sync::{Arc, Mutex};

use crate::ddp::worker::DdpWorker;
use crate::metric::store::EventStoreClient;
use crate::{
    DistributedRuntime, EarlyStoppingStrategyRef, Interrupter, Learner, LearningComponentsTypes,
    SupervisedLearningStrategy, SupervisedTrainingEventProcessor, TrainLoader, TrainingComponents,
    TrainingModel, ValidLoader,
};
use burn_core::data::dataloader::split::split_dataloader;
use burn_core::tensor::Device;

#[derive(Clone)]
pub(crate) struct WorkerComponents {
    /// The total number of epochs
    pub num_epochs: usize,
    /// Enables gradients accumulation.
    pub grad_accumulation: Option<usize>,
    /// An [Interupter](Interrupter) that allows aborting the training/evaluation process early.
    pub interrupter: Interrupter,
    /// Cloneable reference to an early stopping strategy.
    pub early_stopping: Option<EarlyStoppingStrategyRef>,
    /// A reference to an [EventStoreClient](EventStoreClient).
    pub event_store: Arc<EventStoreClient>,
}

/// A training strategy for Distributed Data Parallel (DDP) training.
///
/// This strategy manages multiple workers and coordinates cross-device
/// gradient synchronization using the provided [`DistributedRuntime`].
pub struct DdpTrainingStrategy<LC: LearningComponentsTypes> {
    devices: Vec<Device>,
    runtime: Box<dyn DistributedRuntime>,
}
impl<LC: LearningComponentsTypes> DdpTrainingStrategy<LC> {
    pub fn new(devices: Vec<Device>, runtime: Box<dyn DistributedRuntime>) -> Self {
        Self { devices, runtime }
    }
}

impl<LC> SupervisedLearningStrategy<LC> for DdpTrainingStrategy<LC>
where
    LC: LearningComponentsTypes + Send + 'static,
{
    fn fit(
        &self,
        training_components: TrainingComponents<LC>,
        learner: Learner<LC>,
        dataloader_train: TrainLoader<LC>,
        dataloader_valid: ValidLoader<LC>,
        starting_epoch: usize,
    ) -> (TrainingModel<LC>, SupervisedTrainingEventProcessor<LC>) {
        // The reference model is always on the first device provided.
        let main_device = self.devices.first().unwrap();
        // One worker per device, so we use a fixed device strategy
        // for each (worker) data loader. This matches the expected device on the worker, so we
        // don't have to move the data between devices.
        let mut dataloaders_train = split_dataloader(dataloader_train, &self.devices);
        let dataloader_valid = dataloader_valid.to_device(&main_device.clone().inner());

        let main_device = self.devices[0].clone();
        let peer_count = self.devices.len();
        let event_processor = Arc::new(Mutex::new(training_components.event_processor));

        let interrupter = training_components.interrupter;
        let worker_components = WorkerComponents {
            num_epochs: training_components.num_epochs,
            grad_accumulation: training_components.grad_accumulation,
            interrupter: interrupter.clone(),
            early_stopping: training_components.early_stopping,
            event_store: training_components.event_store,
        };

        self.runtime.start();

        // Start worker for main device
        // First training dataloader corresponds to main device
        let main_handle = DdpWorker::<LC>::start(
            main_device.clone(),
            learner.clone(),
            event_processor.clone(),
            worker_components.clone(),
            training_components.checkpointer,
            dataloaders_train.remove(0),
            Some(dataloader_valid),
            starting_epoch,
            peer_count,
            true,
        );

        // Spawn other workers for the other devices, starting with peer id 1
        let mut secondary_workers = vec![];
        for device in &self.devices[1..] {
            let handle = DdpWorker::<LC>::start(
                device.clone(),
                learner.clone(),
                event_processor.clone(),
                worker_components.clone(),
                None,
                dataloaders_train.remove(0),
                None,
                starting_epoch,
                peer_count,
                false,
            );

            secondary_workers.push(handle);
        }

        // Wait for all devices to finish
        for worker in secondary_workers {
            worker
                .join()
                .expect("Distributed data parallel worker failed");
        }

        self.runtime.close();

        // Main worker had the event processor
        let model = main_handle
            .join()
            .expect("Distributed data parallel main worker failed");

        if interrupter.should_stop() {
            let reason = interrupter
                .get_message()
                .unwrap_or(String::from("Reason unknown"));
            log::info!("Training interrupted: {reason}");
        }
        let Ok(event_processor) = Arc::try_unwrap(event_processor) else {
            panic!("Event processor still held!");
        };
        let Ok(event_processor) = event_processor.into_inner() else {
            panic!("Event processor lock poisoned");
        };
        (model, event_processor)
    }
}
