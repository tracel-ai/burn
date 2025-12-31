use crate::CloneEarlyStoppingStrategy;
use crate::checkpoint::{
    AsyncCheckpointer, Checkpointer, CheckpointingAction, CheckpointingStrategy,
};
use crate::components::{LearningComponentsTypes, TrainBackend};
use crate::metric::store::EventStoreClient;
use crate::{LearningComponentsMarker, ParadigmComponentsTypes, TrainingResult};
use burn_core::module::{AutodiffModule, Module};
use burn_core::prelude::Backend;
use burn_core::tensor::Device;
use burn_core::tensor::backend::AutodiffBackend;
use burn_optim::Optimizer;
use burn_optim::lr_scheduler::LrScheduler;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The record of the learning model.
pub type LearnerModelRecord<LC> =
    <<LC as LearningComponentsTypes>::Model as Module<TrainBackend<LC>>>::Record;
/// The record of the optimizer.
pub type LearnerOptimizerRecord<LC> = <<LC as LearningComponentsTypes>::Optimizer as Optimizer<
    <LC as LearningComponentsTypes>::Model,
    TrainBackend<LC>,
>>::Record;
/// The record of the LR scheduler.
pub type LearnerSchedulerRecord<LC> =
    <<LC as LearningComponentsTypes>::LrScheduler as LrScheduler>::Record<TrainBackend<LC>>;

/// Provides the `run` function for any learning paradigm.
pub trait LearningParadigm<LC>
where
    LC: LearningComponentsTypes,
{
    /// Executes the learning paradigm using the provided learner.
    ///
    /// This method drives the full learning process (e.g. training loop, validation,
    /// checkpointing) and returns the final result.
    fn run(self, learner: Learner<LC>) -> TrainingResult<LC::InnerModel>;
}

/// Learner struct encapsulating all components necessary to train a Neural Network model.
#[derive(Clone)]
pub struct Learner<LC: LearningComponentsTypes> {
    /// The neural network model.
    pub model: LC::Model,
    /// The optimizer.
    pub optim: LC::Optimizer,
    /// The learning rate scheduler.
    pub lr_scheduler: LC::LrScheduler,
}

impl<B, LR, M, O> Learner<LearningComponentsMarker<B, LR, M, O>>
where
    B: AutodiffBackend,
    LR: LrScheduler + 'static,
    M: AutodiffModule<B> + core::fmt::Display + 'static,
    O: Optimizer<M, B> + 'static,
{
    /// Create a learner.
    pub fn new(model: M, optim: O, lr_scheduler: LR) -> Self {
        Self {
            model,
            optim,
            lr_scheduler,
        }
    }
}

impl<LC: LearningComponentsTypes> Learner<LC> {
    /// Load the module state from a [record](LearnerModelRecord<LC>).
    pub fn load_model_record(&mut self, record: LearnerModelRecord<LC>) {
        self.model = self.model.clone().load_record(record);
    }

    /// Load the state of the learner's optimizer as a [record](LearnerOptimizerRecord<LC>).
    pub fn load_optim_record(&mut self, record: LearnerOptimizerRecord<LC>) {
        self.optim = self.optim.clone().load_record(record);
    }

    /// Load the state of the learner's scheduler as a [record](LearnerSchedulerRecord<LC>).
    pub fn load_scheduler_record(&mut self, record: LearnerSchedulerRecord<LC>) {
        self.lr_scheduler = self.lr_scheduler.clone().load_record(record);
    }

    /// Fork the learner's model to the given device.
    pub fn fork(self, device: &<TrainBackend<LC> as Backend>::Device) -> Self {
        let model = self.model.fork(device);
        Self {
            model,
            optim: self.optim,
            lr_scheduler: self.lr_scheduler,
        }
    }
}

#[derive(new)]
/// Used to create, delete, or load checkpoints of the training process.
pub struct LearningCheckpointer<LC: LearningComponentsTypes, PC: ParadigmComponentsTypes> {
    model: AsyncCheckpointer<LearnerModelRecord<LC>, LC::Backend>,
    optim: AsyncCheckpointer<LearnerOptimizerRecord<LC>, LC::Backend>,
    lr_scheduler: AsyncCheckpointer<LearnerSchedulerRecord<LC>, LC::Backend>,
    strategy: PC::CheckpointerStrategy,
}

impl<LC: LearningComponentsTypes, PC: ParadigmComponentsTypes> LearningCheckpointer<LC, PC> {
    /// Create checkpoint for the training process.
    pub fn checkpoint(&mut self, learner: &Learner<LC>, epoch: usize, store: &EventStoreClient) {
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
                        .save(epoch, learner.model.clone().into_record())
                        .expect("Can save model checkpoint.");
                    self.optim
                        .save(epoch, learner.optim.to_record())
                        .expect("Can save optimizer checkpoint.");
                    self.lr_scheduler
                        .save(epoch, learner.lr_scheduler.to_record())
                        .expect("Can save learning rate scheduler checkpoint.");
                }
            }
        }
    }

    /// Load a training checkpoint.
    pub fn load_checkpoint(
        &self,
        mut learner: Learner<LC>,
        device: &Device<LC::Backend>,
        epoch: usize,
    ) -> Learner<LC> {
        let record = self
            .model
            .restore(epoch, device)
            .expect("Can load model checkpoint.");
        learner.load_model_record(record);

        let record = self
            .optim
            .restore(epoch, device)
            .expect("Can load optimizer checkpoint.");
        learner.load_optim_record(record);

        let record = self
            .lr_scheduler
            .restore(epoch, device)
            .expect("Can load learning rate scheduler checkpoint.");
        learner.load_scheduler_record(record);

        learner
    }
}

/// Cloneable reference to an early stopping strategy
pub(crate) type EarlyStoppingStrategyRef = Box<dyn CloneEarlyStoppingStrategy>;

#[derive(Clone, Default)]
/// A handle that allows aborting the training/evaluation process early.
pub struct Interrupter {
    state: Arc<AtomicBool>,
    message: Arc<Mutex<Option<String>>>,
}

impl Interrupter {
    /// Create a new instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Notify the learner that it should stop.
    /// # Arguments
    /// * `reason` - A string describing the reason the training was stopped.
    pub fn stop(&self, reason: Option<&str>) {
        self.state.store(true, Ordering::Relaxed);
        reason.inspect(|r| {
            let mut message = self.message.lock().unwrap();
            *message = Some(String::from(*r));
        });
    }

    /// Reset the interrupter.
    pub fn reset(&self) {
        self.state.store(false, Ordering::Relaxed);
    }

    /// True if .stop() has been called.
    pub fn should_stop(&self) -> bool {
        self.state.load(Ordering::Relaxed)
    }

    /// Get the message associated with the interrupt.
    pub fn get_message(&self) -> Option<String> {
        let message = self.message.lock().unwrap();
        message.clone()
    }
}
