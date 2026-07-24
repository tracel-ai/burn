use core::panic;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use crate::ddp::worker::DdpWorker;
use crate::metric::store::EventStoreClient;
use crate::{
    EarlyStoppingStrategyRef, Interrupter, Learner, LearnerModel, SupervisedLearningStrategy,
    SupervisedTrainingEventProcessor, TrainLoader, TrainingComponents, ValidLoader,
};
use burn_core::data::dataloader::split::split_dataloader;
use burn_core::tensor::Device;
use burn_core::tensor::distributed::DistributedContext;

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
    /// The total number of items in the training dataset.
    pub train_total_items: usize,
    /// The total number of items in the validation dataset.
    pub valid_total_items: usize,
}

/// A training strategy for Distributed Data Parallel (DDP) training.
///
/// This strategy manages multiple workers and coordinates cross-device
/// gradient synchronization using the provided [`DistributedContext`].
pub struct DdpTrainingStrategy {
    devices: Vec<Device>,
    /// Kept alive to anchor the lifetime of the underlying distributed server.
    /// Spawns communication servers on creation, automatically tears them down on drop.
    _context: DistributedContext,
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic in distributed data parallel worker".to_string()
    }
}

impl DdpTrainingStrategy {
    pub fn new(devices: Vec<Device>, context: DistributedContext) -> Self {
        Self {
            devices,
            _context: context,
        }
    }
}

impl<M: LearnerModel> SupervisedLearningStrategy<M> for DdpTrainingStrategy {
    fn fit(
        &self,
        training_components: TrainingComponents<M>,
        learner: Learner<M>,
        dataloader_train: TrainLoader<M>,
        dataloader_valid: ValidLoader<M>,
        starting_epoch: usize,
    ) -> (M, SupervisedTrainingEventProcessor<M>) {
        // The reference model is always on the first device provided.
        let main_device = self.devices.first().unwrap();
        let train_total_items = dataloader_train.num_items();
        let valid_total_items = dataloader_valid.num_items();
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
            train_total_items,
            valid_total_items,
        };

        // Start worker for main device
        // First training dataloader corresponds to main device
        let main_handle = DdpWorker::<M>::start(
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
            let handle = DdpWorker::<M>::start(
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

        const MAIN_ID: usize = usize::MAX;
        let (result_tx, result_rx) = mpsc::channel();

        for (id, worker) in secondary_workers.into_iter().enumerate() {
            let tx = result_tx.clone();
            thread::spawn(move || {
                tx.send((id, worker.join())).ok();
            });
        }
        {
            let tx = result_tx.clone();
            thread::spawn(move || {
                tx.send((MAIN_ID, main_handle.join())).ok();
            });
        }
        drop(result_tx);

        let mut main_model = None;
        for _ in 0..peer_count {
            match result_rx
                .recv()
                .expect("worker reaper thread disconnected unexpectedly")
            {
                (MAIN_ID, Ok(model)) => main_model = Some(model),
                (id, Err(payload)) => {
                    let msg = panic_message(payload.as_ref());
                    if id == MAIN_ID {
                        panic!("Distributed data parallel main worker failed: {msg}");
                    } else {
                        panic!("Distributed data parallel worker {id} failed: {msg}");
                    }
                }
                (_, Ok(_)) => {}
            }
        }
        // Main worker had the event processor
        let model = main_model.expect("main worker should have produced a model");

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
