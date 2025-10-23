use crate::checkpoint::{Checkpointer, CheckpointingAction, CheckpointingStrategy};
use crate::components::LearnerComponentTypes;
use crate::metric::store::EventStoreClient;
use crate::{CloneEarlyStoppingStrategy, LearnerSummaryConfig, LearningStrategy};
use burn_core::module::Module;
use burn_core::tensor::Device;
use burn_optim::Optimizer;
use burn_optim::lr_scheduler::LrScheduler;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Learner struct encapsulating all components necessary to train a Neural Network model.
///
/// To create a learner, use the [builder](crate::learner::LearnerBuilder) struct.
pub struct Learner<LC: LearnerComponentTypes> {
    pub(crate) model: LC::Model,
    pub(crate) optim: LC::Optimizer,
    pub(crate) lr_scheduler: LC::LrScheduler,
    pub(crate) num_epochs: usize,
    pub(crate) checkpoint: Option<usize>,
    pub(crate) grad_accumulation: Option<usize>,
    pub(crate) checkpointer: Option<LearnerCheckpointer<LC>>,
    pub(crate) learning_strategy: LearningStrategy<LC::Backend, LC>,
    pub(crate) interrupter: Interrupter,
    pub(crate) early_stopping: Option<EarlyStoppingStrategyRef>,
    pub(crate) event_processor: LC::EventProcessor,
    pub(crate) event_store: Arc<EventStoreClient>,
    pub(crate) summary: Option<LearnerSummaryConfig>,
}

/// Cloneable reference to an early stopping strategy
pub(crate) type EarlyStoppingStrategyRef = Box<dyn CloneEarlyStoppingStrategy>;

#[derive(new)]
/// Object used to create, delete, or load checkpoints of the training process.
pub struct LearnerCheckpointer<LC: LearnerComponentTypes> {
    model: LC::CheckpointerModel,
    optim: LC::CheckpointerOptimizer,
    lr_scheduler: LC::CheckpointerLrScheduler,
    strategy: LC::CheckpointerStrategy,
}

impl<LC: LearnerComponentTypes> LearnerCheckpointer<LC> {
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
                    self.model
                        .delete(epoch)
                        .expect("Can delete model checkpoint.");
                    self.optim
                        .delete(epoch)
                        .expect("Can delete optimizer checkpoint.");
                    self.lr_scheduler
                        .delete(epoch)
                        .expect("Can delete learning rate scheduler checkpoint.");
                }
                CheckpointingAction::Save => {
                    self.model
                        .save(epoch, model.clone().into_record())
                        .expect("Can save model checkpoint.");
                    self.optim
                        .save(epoch, optim.to_record())
                        .expect("Can save optimizer checkpoint.");
                    self.lr_scheduler
                        .save(epoch, scheduler.to_record())
                        .expect("Can save learning rate scheduler checkpoint.");
                }
            }
        }
    }

    pub(crate) fn load_checkpoint(
        &self,
        model: LC::Model,
        optim: LC::Optimizer,
        scheduler: LC::LrScheduler,
        device: &Device<LC::Backend>,
        epoch: usize,
    ) -> (LC::Model, LC::Optimizer, LC::LrScheduler) {
        let record = self
            .model
            .restore(epoch, device)
            .expect("Can load model checkpoint.");
        let model = model.load_record(record);

        let record = self
            .optim
            .restore(epoch, device)
            .expect("Can load optimizer checkpoint.");
        let optim = optim.load_record(record);

        let record = self
            .lr_scheduler
            .restore(epoch, device)
            .expect("Can load learning rate scheduler checkpoint.");
        let scheduler = scheduler.load_record(record);

        (model, optim, scheduler)
    }
}

#[derive(Clone, Default)]
/// A handle that allows aborting the training/evaluation process early.
pub struct Interrupter {
    state: Arc<AtomicBool>,
}

impl Interrupter {
    /// Create a new instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Notify the learner that it should stop.
    pub fn stop(&self) {
        self.state.store(true, Ordering::Relaxed);
    }

    /// Reset the interrupter.
    pub fn reset(&self) {
        self.state.store(false, Ordering::Relaxed);
    }

    /// True if .stop() has been called.
    pub fn should_stop(&self) -> bool {
        self.state.load(Ordering::Relaxed)
    }
}
