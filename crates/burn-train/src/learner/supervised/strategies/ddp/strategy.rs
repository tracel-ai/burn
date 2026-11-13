use std::sync::{Arc, Mutex, PoisonError, mpsc};

use crate::ddp::worker::DdpWorker;
use crate::metric::store::EventStoreClient;
use crate::train::WorkerFailure;
use crate::{
    EarlyStoppingStrategyRef, Interrupter, Learner, LearnerModel, SupervisedLearningStrategy,
    SupervisedTrainingEventProcessor, TrainLoader, TrainingComponents, TrainingError, ValidLoader,
};
use burn_core::data::dataloader::split::split_dataloader;
use burn_core::tensor::Device;
use burn_core::tensor::distributed::DistributedContext;

use super::barrier::WorkerBarrier;
use super::processor::WorkerEventProcessor;
use super::step::StepCoordinator;

pub(crate) type DdpEventProcessor<M> = WorkerEventProcessor<SupervisedTrainingEventProcessor<M>>;

#[derive(Clone)]
pub(crate) struct WorkerComponents {
    /// The total number of epochs.
    pub num_epochs: usize,
    /// Enables gradients accumulation.
    pub grad_accumulation: Option<usize>,
    /// An [Interrupter] that allows aborting the training/evaluation process early.
    pub interrupter: Interrupter,
    /// Cloneable reference to an early stopping strategy.
    pub early_stopping: Option<EarlyStoppingStrategyRef>,
    /// A reference to an [EventStoreClient].
    pub event_store: Arc<EventStoreClient>,
    /// All workers finish training before validation, and validation before the next epoch.
    pub epoch_barrier: Arc<WorkerBarrier>,
    /// Coordinates batch availability and stop decisions before collective backward passes.
    pub steps: Arc<StepCoordinator>,
}

impl WorkerComponents {
    /// Releases every worker waiting for a peer that failed.
    fn abort(&self) {
        self.epoch_barrier.abort();
        self.steps.abort();
    }
}

/// A training strategy for Distributed Data Parallel (DDP) training.
///
/// This strategy manages multiple workers and coordinates cross-device
/// gradient synchronization using the provided [`DistributedContext`].
pub struct DdpTrainingStrategy {
    devices: Vec<Device>,
    /// Anchors the lifetime of the underlying distributed server, which it tears down on drop.
    /// Dropped early when a worker fails, so peers waiting in a collective for its gradients fail
    /// instead of waiting forever.
    context: Mutex<Option<DistributedContext>>,
}

impl DdpTrainingStrategy {
    pub fn new(devices: Vec<Device>, context: DistributedContext) -> Self {
        Self {
            devices,
            context: Mutex::new(Some(context)),
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
        // One worker per device, so we use a fixed device strategy
        // for each (worker) data loader. This matches the expected device on the worker, so we
        // don't have to move the data between devices.
        let mut dataloaders_train = split_dataloader(dataloader_train, &self.devices);
        let dataloader_valid = dataloader_valid.to_device(&main_device.clone().inner());

        let main_device = self.devices[0].clone();
        let peer_count = self.devices.len();
        let totals = dataloaders_train
            .iter()
            .map(|d| d.expected_num_items())
            .collect();
        let event_processor = Arc::new(Mutex::new(WorkerEventProcessor::new(
            training_components.event_processor,
            totals,
            training_components.interrupter.clone(),
        )));

        let interrupter = training_components.interrupter;
        let worker_components = WorkerComponents {
            num_epochs: training_components.num_epochs,
            grad_accumulation: training_components.grad_accumulation,
            interrupter: interrupter.clone(),
            early_stopping: training_components.early_stopping,
            event_store: training_components.event_store,
            epoch_barrier: Arc::new(WorkerBarrier::new(peer_count)),
            steps: Arc::new(StepCoordinator::new(peer_count)),
        };
        let (outcome_tx, outcome_rx) = mpsc::channel();

        // Start worker for main device
        // First training dataloader corresponds to main device
        let mut handles = vec![DdpWorker::<M>::start(
            main_device.clone(),
            learner.clone(),
            event_processor.clone(),
            worker_components.clone(),
            training_components.checkpointer,
            dataloaders_train.remove(0),
            Some(dataloader_valid),
            starting_epoch,
            0,
            true,
            outcome_tx.clone(),
        )];

        // Spawn other workers for the other devices, starting with peer id 1
        for (index, device) in self.devices[1..].iter().enumerate() {
            handles.push(DdpWorker::<M>::start(
                device.clone(),
                learner.clone(),
                event_processor.clone(),
                worker_components.clone(),
                None,
                dataloaders_train.remove(0),
                None,
                starting_epoch,
                index + 1,
                false,
                outcome_tx.clone(),
            ));
        }
        drop(outcome_tx);

        let mut main_model = None;
        // Outcomes arrive as workers finish, so a failure is handled while its peers still wait.
        for outcome in outcome_rx.iter().take(peer_count) {
            if let Some(message) = outcome.failure {
                // The first failure is the cause: the interrupter only logs the ones it triggers.
                interrupter.fail(TrainingError::Workers(vec![WorkerFailure {
                    device_id: outcome.worker_id,
                    message,
                }]));
                worker_components.abort();
                // Recorded before closing, so a peer whose backward then skips the sync sees it.
                drop(
                    self.context
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .take(),
                );
            }
            if outcome.worker_id == 0 {
                main_model = Some(outcome.model);
            }
        }
        for handle in handles {
            // Every worker reported, so each thread has finished or is about to.
            handle.join().ok();
        }
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
        // A worker that panicked while reporting poisons the lock; its failure is already recorded.
        let event_processor = event_processor
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);
        (model, event_processor.into_inner())
    }
}
