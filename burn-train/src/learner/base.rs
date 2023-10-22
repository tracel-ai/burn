use crate::checkpoint::{Checkpointer, CheckpointingAction, CheckpointingStrategy};
use crate::components::LearnerComponents;
use crate::learner::EarlyStoppingStrategy;
use crate::metric::store::EventStoreClient;
use burn_core::lr_scheduler::LrScheduler;
use burn_core::module::Module;
use burn_core::optim::Optimizer;
use burn_core::tensor::backend::Backend;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Learner struct encapsulating all components necessary to train a Neural Network model.
///
/// To create a learner, use the [builder](crate::learner::LearnerBuilder) struct.
pub struct Learner<LC: LearnerComponents> {
    pub(crate) model: LC::Model,
    pub(crate) optim: LC::Optimizer,
    pub(crate) lr_scheduler: LC::LrScheduler,
    pub(crate) num_epochs: usize,
    pub(crate) checkpoint: Option<usize>,
    pub(crate) grad_accumulation: Option<usize>,
    pub(crate) checkpointer: Option<LearnerCheckpointer<LC>>,
    pub(crate) devices: Vec<<LC::Backend as Backend>::Device>,
    pub(crate) interrupter: TrainingInterrupter,
    pub(crate) early_stopping: Option<Box<dyn EarlyStoppingStrategy>>,
    pub(crate) event_processor: LC::EventProcessor,
    pub(crate) event_store: Arc<EventStoreClient>,
}

#[derive(new)]
pub(crate) struct LearnerCheckpointer<LC: LearnerComponents> {
    model: LC::CheckpointerModel,
    optim: LC::CheckpointerOptimizer,
    lr_scheduler: LC::CheckpointerLrScheduler,
    strategy: LC::CheckpointerStrategy,
}

impl<LC: LearnerComponents> LearnerCheckpointer<LC> {
    pub(crate) fn checkpoint(
        &mut self,
        model: &LC::Model,
        optim: &LC::Optimizer,
        scheduler: &LC::LrScheduler,
        epoch: usize,
        store: &EventStoreClient,
    ) {
        let actions = self.strategy.checkpointing(epoch, store);

        for action in actions {
            match action {
                CheckpointingAction::Delete(epoch) => {
                    self.model.delete(epoch).unwrap();
                    self.optim.delete(epoch).unwrap();
                    self.lr_scheduler.delete(epoch).unwrap();
                }
                CheckpointingAction::Save => {
                    self.model.save(epoch, model.clone().into_record()).unwrap();
                    self.optim.save(epoch, optim.to_record()).unwrap();
                    self.lr_scheduler
                        .save(epoch, scheduler.to_record())
                        .unwrap();
                }
            }
        }
    }

    pub(crate) fn load_checkpoint(
        &self,
        model: LC::Model,
        optim: LC::Optimizer,
        scheduler: LC::LrScheduler,
        epoch: usize,
    ) -> (LC::Model, LC::Optimizer, LC::LrScheduler) {
        let record = self.model.restore(epoch).unwrap();
        let model = model.load_record(record);

        let record = self.optim.restore(epoch).unwrap();
        let optim = optim.load_record(record);

        let record = self.lr_scheduler.restore(epoch).unwrap();
        let scheduler = scheduler.load_record(record);

        (model, optim, scheduler)
    }
}

#[derive(Clone, Default)]
/// A handle that allows aborting the training process early.
pub struct TrainingInterrupter {
    state: Arc<AtomicBool>,
}

impl TrainingInterrupter {
    /// Create a new instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Notify the learner that it should stop.
    pub fn stop(&self) {
        self.state.store(true, Ordering::Relaxed);
    }

    /// True if .stop() has been called.
    pub fn should_stop(&self) -> bool {
        self.state.load(Ordering::Relaxed)
    }
}
