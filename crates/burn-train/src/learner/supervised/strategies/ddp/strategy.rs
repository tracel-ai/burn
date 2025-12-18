use core::panic;
use std::sync::{Arc, Mutex};

use burn_collective::CollectiveConfig;
use burn_core::tensor::Device;

use crate::ddp::worker::DdpWorkerV2;
use crate::metric::store::EventStoreClient;
use crate::{
    EarlyStoppingStrategyRef, Interrupter, Learner, SupervisedLearningComponentsTypes,
    SupervisedLearningStrategy, TrainBackend, TrainLoader, TrainingComponents, ValidLoader,
};
use burn_core::data::dataloader::split::split_dataloader;

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

pub struct DdpTrainingStrategy<SC: SupervisedLearningComponentsTypes> {
    devices: Vec<Device<TrainBackend<SC::LC>>>,
    config: CollectiveConfig,
}
impl<SC: SupervisedLearningComponentsTypes> DdpTrainingStrategy<SC> {
    pub fn new(devices: Vec<Device<TrainBackend<SC::LC>>>, config: CollectiveConfig) -> Self {
        let config = config.with_num_devices(devices.len());
        Self { devices, config }
    }
}

impl<SC: SupervisedLearningComponentsTypes + Send + 'static> SupervisedLearningStrategy<SC>
    for DdpTrainingStrategy<SC>
{
    fn fit(
        &self,
        training_components: TrainingComponents<SC>,
        learner: Learner<SC::LC>,
        dataloader_train: TrainLoader<SC::LC, SC::LD>,
        dataloader_valid: ValidLoader<SC::LC, SC::LD>,
        starting_epoch: usize,
    ) -> (
        SC::Model,
        <SC::PC as crate::ParadigmComponentsTypes>::EventProcessor,
    ) {
        // The reference model is always on the first device provided.
        let main_device = self.devices.first().unwrap();
        // One worker per device, so we use a fixed device strategy
        // for each (worker) data loader. This matches the expected device on the worker, so we
        // don't have to move the data between devices.
        let mut dataloaders_train = split_dataloader(dataloader_train, &self.devices);
        let dataloader_valid = dataloader_valid.to_device(main_device);

        let main_device = self.devices[0].clone();
        let peer_count = self.devices.len();
        let event_processor = Arc::new(Mutex::new(training_components.event_processor));

        let worker_components = WorkerComponents {
            num_epochs: training_components.num_epochs,
            grad_accumulation: training_components.grad_accumulation,
            interrupter: training_components.interrupter,
            early_stopping: training_components.early_stopping,
            event_store: training_components.event_store,
        };

        // Start worker for main device
        // First training dataloader corresponds to main device
        let main_handle = DdpWorkerV2::<SC>::start(
            0.into(),
            main_device,
            learner.clone(),
            event_processor.clone(),
            worker_components.clone(),
            training_components.checkpointer,
            dataloaders_train.remove(0),
            Some(dataloader_valid),
            self.config.clone(),
            starting_epoch,
            peer_count,
            true,
        );

        // Spawn other workers for the other devices, starting with peer id 1
        let mut peer_id = 1;
        let mut secondary_workers = vec![];
        for device in &self.devices[1..] {
            let handle = DdpWorkerV2::<SC>::start(
                peer_id.into(),
                device.clone(),
                learner.clone(),
                event_processor.clone(),
                worker_components.clone(),
                None,
                dataloaders_train.remove(0),
                None,
                self.config.clone(),
                starting_epoch,
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
