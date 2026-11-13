use crate::ddp::epoch::{DdpTrainEpoch, DdpValidEpoch};
use crate::ddp::strategy::{DdpEventProcessor, WorkerComponents};
use crate::single::TrainingLoop;
use crate::train::panic_message;
use crate::{Learner, LearnerModel, LearningCheckpointer, TrainLoader, ValidLoader};
use burn_core::tensor::Device;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// A worker runs the model, syncing gradients using collective operations.
/// Event processing and validation is optional too.
pub(crate) struct DdpWorker<M: LearnerModel> {
    device: Device,
    learner: Learner<M>,
    event_processor: Arc<Mutex<DdpEventProcessor<M>>>,
    components: WorkerComponents,
    checkpointer: Option<LearningCheckpointer<M>>,
    dataloader_train: TrainLoader<M>,
    dataloader_valid: Option<ValidLoader<M>>,
    starting_epoch: usize,
    worker_id: usize,
    is_main: bool,
}

/// How a worker finished.
pub(crate) struct WorkerOutcome<M> {
    pub worker_id: usize,
    /// The model after the last completed optimizer step.
    pub model: M,
    /// The panic message, if the worker failed.
    pub failure: Option<String>,
}

impl<M: LearnerModel> DdpWorker<M> {
    /// Starts a worker that runs the model in a data distributed parallel, reporting how it
    /// finished to `outcomes`.
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        device: Device,
        learner: Learner<M>,
        event_processor: Arc<Mutex<DdpEventProcessor<M>>>,
        components: WorkerComponents,
        checkpointer: Option<LearningCheckpointer<M>>,
        dataloader_train: TrainLoader<M>,
        dataloader_valid: Option<ValidLoader<M>>,
        starting_epoch: usize,
        worker_id: usize,
        is_main: bool,
        outcomes: Sender<WorkerOutcome<M>>,
    ) -> JoinHandle<()> {
        let worker = Self {
            device,
            learner,
            event_processor,
            components,
            checkpointer,
            dataloader_train,
            dataloader_valid,
            starting_epoch,
            worker_id,
            is_main,
        };

        std::thread::spawn(move || {
            // The receiver only leaves once every worker reported.
            outcomes.send(worker.fit()).ok();
        })
    }

    /// Fits the model, catching a panic so its peers can be released.
    fn fit(mut self) -> WorkerOutcome<M> {
        let result = catch_unwind(AssertUnwindSafe(|| self.run()));
        WorkerOutcome {
            worker_id: self.worker_id,
            model: self.learner.model(),
            failure: result.err().map(|payload| panic_message(payload.as_ref())),
        }
    }

    fn run(&mut self) {
        let num_epochs = self.components.num_epochs;
        let interrupter = self.components.interrupter.clone();

        // Changed the train epoch to keep the dataloaders
        let epoch_train = DdpTrainEpoch::<M>::new(
            self.dataloader_train.clone(),
            self.components.grad_accumulation,
            self.components.steps.clone(),
        );
        let epoch_valid = self
            .dataloader_valid
            .clone()
            .map(|dataloader| DdpValidEpoch::<M>::new(dataloader));
        self.learner.fork(&self.device);
        self.learner.grad_sharded();

        for training_progress in TrainingLoop::new(self.starting_epoch, num_epochs) {
            let epoch = training_progress.items_processed;

            if self.is_main {
                self.event_processor.lock().unwrap().start_epoch(epoch);
            }

            // An aborted barrier means a peer failed: no further collective can complete.
            if self.components.epoch_barrier.wait().is_err() {
                return;
            }

            epoch_train.run(
                &mut self.learner,
                &training_progress,
                self.event_processor.clone(),
                &interrupter,
                self.worker_id,
                &self.components.epoch_barrier,
            );
            if self.components.epoch_barrier.wait().is_err() {
                return;
            }
            if self.is_main {
                self.event_processor.lock().unwrap().finish_train(epoch);
            }
            // Only the main worker validates; all peers wait at the epoch barrier below.
            if let Some(runner) = &epoch_valid
                && !interrupter.should_stop()
            {
                runner.run(
                    &self.learner.model(),
                    &training_progress,
                    &self.event_processor,
                    &interrupter,
                );
            }

            if self.is_main {
                self.event_processor.lock().unwrap().end_epoch(epoch);
                let mut stop = interrupter.should_stop();
                if !stop {
                    if let Some(checkpointer) = &mut self.checkpointer {
                        checkpointer.checkpoint(&self.learner, epoch, &self.components.event_store);
                    }
                    if let Some(early_stopping) = &mut self.components.early_stopping {
                        stop = early_stopping.should_stop(epoch, &self.components.event_store);
                    }
                }
                self.event_processor.lock().unwrap().stop_after_epoch = stop;
            }
            if self.components.epoch_barrier.wait().is_err() {
                return;
            }
            // Every worker makes the same epoch decision, even if a stop request races the barrier.
            if self.event_processor.lock().unwrap().stop_after_epoch {
                break;
            }
        }
    }
}
