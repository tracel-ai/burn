use crate::LearningComponentsMarker;
use crate::checkpoint::{
    AsyncCheckpointer, Checkpointer, CheckpointingAction, CheckpointingStrategy,
};
use crate::components::{LearningComponentsTypes, TrainingBackend};
use crate::metric::store::EventStoreClient;
use crate::{
    CloneEarlyStoppingStrategy, InferenceStep, TrainOutput, TrainStep, TrainingModelInput,
    TrainingModelOutput,
};
use burn_core::module::{AutodiffModule, Module};
use burn_core::prelude::Backend;
use burn_core::tensor::Device;
use burn_core::tensor::backend::{AutodiffBackend, PeerId, ReduceOperation};
use burn_optim::lr_scheduler::LrScheduler;
use burn_optim::{GradientsParams, MultiGradientsParams, Optimizer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The record of the learner's model.
pub type LearnerModelRecord<LC> =
    <<LC as LearningComponentsTypes>::TrainingModel as Module<TrainingBackend<LC>>>::Record;
/// The record of the optimizer.
pub type LearnerOptimizerRecord<LC> = <<LC as LearningComponentsTypes>::Optimizer as Optimizer<
    <LC as LearningComponentsTypes>::TrainingModel,
    TrainingBackend<LC>,
>>::Record;
/// The record of the LR scheduler.
pub type LearnerSchedulerRecord<LC> =
    <<LC as LearningComponentsTypes>::LrScheduler as LrScheduler>::Record<TrainingBackend<LC>>;

/// Learner struct encapsulating all components necessary to train a Neural Network model.
pub struct Learner<LC: LearningComponentsTypes> {
    pub(crate) model: LC::TrainingModel,
    optim: LC::Optimizer,
    lr_scheduler: LC::LrScheduler,
    lr: f64,
}

impl<LC: LearningComponentsTypes> Clone for Learner<LC> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            optim: self.optim.clone(),
            lr_scheduler: self.lr_scheduler.clone(),
            lr: self.lr,
        }
    }
}

impl<B, LR, M, O> Learner<LearningComponentsMarker<B, LR, M, O>>
where
    B: AutodiffBackend,
    LR: LrScheduler + 'static,
    M: TrainStep + AutodiffModule<B> + core::fmt::Display + 'static,
    M::InnerModule: InferenceStep,
    O: Optimizer<M, B> + 'static,
{
    /// Create a learner.
    pub fn new(model: M, optim: O, lr_scheduler: LR) -> Self {
        Self {
            model,
            optim,
            lr_scheduler,
            lr: 0.0,
        }
    }
}

impl<LC: LearningComponentsTypes> Learner<LC> {
    /// Fork the learner's model to the given device.
    pub fn fork(&mut self, device: &<TrainingBackend<LC> as Backend>::Device) {
        self.model = self.model().fork(device);
    }

    /// Mark the model as sharded across multiple devices.
    ///
    /// # Arguments
    ///
    /// * `peer_id` - The device's [PeerId](PeerId).
    /// * `op` - The reduce operation.
    pub fn grad_sharded(&mut self, peer_id: PeerId, op: ReduceOperation) {
        self.model = self.model.clone().grad_sharded(peer_id, op);
    }

    /// Returns the current model.
    pub fn model(&self) -> LC::TrainingModel {
        self.model.clone()
    }

    /// Returns the current learning rate.
    pub fn lr_current(&self) -> f64 {
        self.lr
    }

    /// Executes a step of the learning rate scheduler.
    pub fn lr_step(&mut self) {
        self.lr = self.lr_scheduler.step();
    }

    /// Runs a step of the model for training, which executes the forward and backward passes.
    ///
    /// # Arguments
    ///
    /// * `item` - The input for the model.
    ///
    /// # Returns
    ///
    /// The output containing the model output and the gradients.
    pub fn train_step(&self, item: TrainingModelInput<LC>) -> TrainOutput<TrainingModelOutput<LC>> {
        self.model.step(item)
    }

    /// Optimize the current module with the provided gradients and learning rate.
    ///
    /// # Arguments
    ///
    /// * `optim`: Optimizer used for learning.
    /// * `lr`: The learning rate used for this step.
    /// * `grads`: The gradients of each parameter in the current model.
    pub fn optimizer_step(&mut self, grads: GradientsParams) {
        self.model = self.model().optimize(&mut self.optim, self.lr, grads);
    }

    /// Optimize the current module with the provided gradients and learning rate.
    ///
    /// # Arguments
    ///
    /// * `optim`: Optimizer used for learning.
    /// * `lr`: The learning rate used for this step.
    /// * `grads`: Multiple gradients associated to each parameter in the current model.
    pub fn optimizer_step_multi(&mut self, grads: MultiGradientsParams) {
        self.model = self.model().optimize_multi(&mut self.optim, self.lr, grads);
    }

    /// Load the module state from a [record](LearnerModelRecord<LC>).
    pub fn load_model(&mut self, record: LearnerModelRecord<LC>) {
        self.model = self.model.clone().load_record(record);
    }

    /// Load the state of the learner's optimizer as a [record](LearnerOptimizerRecord<LC>).
    pub fn load_optim(&mut self, record: LearnerOptimizerRecord<LC>) {
        self.optim = self.optim.clone().load_record(record);
    }

    /// Load the state of the learner's scheduler as a [record](LearnerSchedulerRecord<LC>).
    pub fn load_scheduler(&mut self, record: LearnerSchedulerRecord<LC>) {
        self.lr_scheduler = self.lr_scheduler.clone().load_record(record);
    }
}

#[derive(new)]
/// Used to create, delete, or load checkpoints of the training process.
pub struct LearningCheckpointer<LC: LearningComponentsTypes> {
    model: AsyncCheckpointer<LearnerModelRecord<LC>, LC::Backend>,
    optim: AsyncCheckpointer<LearnerOptimizerRecord<LC>, LC::Backend>,
    lr_scheduler: AsyncCheckpointer<LearnerSchedulerRecord<LC>, LC::Backend>,
    strategy: Box<dyn CheckpointingStrategy>,
}

impl<LC: LearningComponentsTypes> LearningCheckpointer<LC> {
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
        learner.load_model(record);

        let record = self
            .optim
            .restore(epoch, device)
            .expect("Can load optimizer checkpoint.");
        learner.load_optim(record);

        let record = self
            .lr_scheduler
            .restore(epoch, device)
            .expect("Can load learning rate scheduler checkpoint.");
        learner.load_scheduler(record);

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
