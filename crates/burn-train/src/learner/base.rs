use crate::checkpoint::{
    AsyncCheckpointer, Checkpointer, CheckpointingAction, CheckpointingStrategy,
};
use crate::metric::store::EventStoreClient;
use crate::{
    CloneEarlyStoppingStrategy, LearnerModel, TrainOutput, TrainStep, TrainingError,
    TrainingModelInput, TrainingModelOutput,
};
use burn_core::store::ModuleRecord;
use burn_core::tensor::Device;
use burn_optim::lr_scheduler::LrSchedulerRecord;
use burn_optim::lr_scheduler::module_lr_scheduler::{ModuleLearningRate, ModuleLrScheduler};
use burn_optim::{GradientsParams, ModuleOptimizer, MultiGradientsParams, OptimizerRecord};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Learner struct encapsulating all components necessary to train a Neural Network model.
pub struct Learner<M: LearnerModel> {
    pub(crate) model: M,
    optim: ModuleOptimizer,
    lr_scheduler: ModuleLrScheduler,
    lr_module: ModuleLearningRate,
}

impl<M: LearnerModel> Clone for Learner<M> {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            optim: self.optim.clone(),
            lr_scheduler: self.lr_scheduler.clone(),
            lr_module: self.lr_module.clone(),
        }
    }
}

impl<M: LearnerModel> Learner<M> {
    /// Create a learner.
    pub fn new(
        model: M,
        optim: ModuleOptimizer,
        lr_scheduler: impl Into<ModuleLrScheduler>,
    ) -> Self {
        Self {
            model,
            optim,
            lr_scheduler: lr_scheduler.into(),
            lr_module: 0.0.into(),
        }
    }
}

impl<M: LearnerModel> Learner<M> {
    /// Fork the learner's model to the given device.
    pub fn fork(&mut self, device: &Device) {
        self.model = self.model().fork(device);
    }

    /// Returns the current model.
    pub fn model(&self) -> M {
        self.model.clone()
    }

    /// Returns the current learning rate.
    pub fn lr_current(&self) -> ModuleLearningRate {
        self.lr_module.clone()
    }

    /// Executes a step of the learning rate scheduler.
    pub fn lr_step(&mut self) {
        self.lr_module = self.lr_scheduler.step();
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
    pub fn train_step(&self, item: TrainingModelInput<M>) -> TrainOutput<TrainingModelOutput<M>> {
        TrainStep::step(&self.model, item)
    }

    /// Optimize the current module with the provided gradients and learning rate.
    ///
    /// # Arguments
    ///
    /// * `optim`: Optimizer used for learning.
    /// * `lr`: The learning rate used for this step.
    /// * `grads`: The gradients of each parameter in the current model.
    pub fn optimizer_step(&mut self, grads: GradientsParams) {
        self.model = self
            .model()
            .optimize(&mut self.optim, self.lr_module.clone(), grads);
    }

    /// Optimize the current module with the provided gradients and learning rate.
    ///
    /// # Arguments
    ///
    /// * `optim`: Optimizer used for learning.
    /// * `lr`: The learning rate used for this step.
    /// * `grads`: Multiple gradients associated to each parameter in the current model.
    pub fn optimizer_step_multi(&mut self, grads: MultiGradientsParams) {
        self.model = self
            .model()
            .optimize_multi(&mut self.optim, self.lr_module.clone(), grads);
    }

    /// Load the module state from a [record](ModuleRecord).
    pub fn load_model(&mut self, record: ModuleRecord) {
        self.model = self.model.clone().load_record(record);
    }

    /// Load the state of the learner's optimizer from a [record](OptimizerRecord).
    ///
    /// No device is needed: the optimizer state is migrated to each parameter's device on the next
    /// step (see [`ModuleOptimizer::load_record`](burn_optim::ModuleOptimizer::load_record)).
    pub fn load_optim(&mut self, record: OptimizerRecord) {
        self.optim = self.optim.clone().load_record(record);
    }

    /// Load the state of the learner's scheduler from a [record](LrSchedulerRecord).
    pub fn load_scheduler(&mut self, record: LrSchedulerRecord) {
        self.lr_scheduler = self.lr_scheduler.clone().load_record(record);
    }
}

/// Used to create, delete, or load checkpoints of the training process.
pub struct LearningCheckpointer<M: LearnerModel> {
    model: AsyncCheckpointer<ModuleRecord>,
    optim: AsyncCheckpointer<OptimizerRecord>,
    lr_scheduler: AsyncCheckpointer<LrSchedulerRecord>,
    strategy: Box<dyn CheckpointingStrategy>,
    _phantom: PhantomData<M>,
}

impl<M: LearnerModel> LearningCheckpointer<M> {
    /// Create a new learning checkpointer.
    pub fn new(
        model: AsyncCheckpointer<ModuleRecord>,
        optim: AsyncCheckpointer<OptimizerRecord>,
        lr_scheduler: AsyncCheckpointer<LrSchedulerRecord>,
        strategy: Box<dyn CheckpointingStrategy>,
    ) -> Self {
        Self {
            model,
            optim,
            lr_scheduler,
            strategy,
            _phantom: PhantomData,
        }
    }

    /// Create checkpoint for the training process.
    pub fn checkpoint(&mut self, learner: &Learner<M>, epoch: usize, store: &EventStoreClient) {
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
    ///
    /// No device is taken: checkpoints are device-free burnpack records (file-backed bytes). On
    /// load, the model keeps the device of the learner's existing parameters, and the optimizer
    /// state is migrated to each parameter's device on the next step. The training device is fixed
    /// earlier, when the learner's model is created/forked.
    pub fn load_checkpoint(&self, mut learner: Learner<M>, epoch: usize) -> Learner<M> {
        let record = self
            .model
            .restore(epoch)
            .expect("Can load model checkpoint.");
        learner.load_model(record);

        let record = self
            .optim
            .restore(epoch)
            .expect("Can load optimizer checkpoint.");
        learner.load_optim(record);

        let record = self
            .lr_scheduler
            .restore(epoch)
            .expect("Can load learning rate scheduler checkpoint.");
        learner.load_scheduler(record);

        learner
    }
}

/// Cloneable reference to an early stopping strategy
pub(crate) type EarlyStoppingStrategyRef = Box<dyn CloneEarlyStoppingStrategy>;

/// A stop made through [`Interrupter::stop`], without an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interruption {
    /// The reason given to [`Interrupter::stop`], if any.
    pub reason: Option<String>,
}

#[derive(Clone, Default)]
/// A handle that allows aborting the training/evaluation process early.
///
/// The process stops either on request, through [`stop`](Self::stop), or because of an error,
/// recorded with [`fail`](Self::fail). An error takes precedence.
pub struct Interrupter {
    state: Arc<AtomicBool>,
    message: Arc<Mutex<Option<String>>>,
    error: Arc<Mutex<Option<Arc<TrainingError>>>>,
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

    /// Stop the process and keep `error` as the cause.
    ///
    /// The first call to this function logs the error and keeps it as the cause of the
    /// interruption. Subsequent calls only log the errors as they are most likely consequences
    /// of the first one.
    pub fn fail(&self, error: impl Into<TrainingError>) {
        let error = error.into();
        log::error!("Stopped by an error: {error}");
        let mut slot = self.error.lock().unwrap();
        if slot.is_none() {
            *slot = Some(Arc::new(error));
        }
        self.state.store(true, Ordering::Relaxed);
    }

    /// [`fail`](Self::fail) with `result`'s error if there is one.
    pub fn fail_on_error<E: Into<TrainingError>>(&self, result: Result<(), E>) {
        if let Err(error) = result {
            self.fail(error);
        }
    }

    /// Reset the interrupter, forgetting any stop and error.
    pub fn reset(&self) {
        self.state.store(false, Ordering::Relaxed);
        self.error.lock().unwrap().take();
    }

    /// True if the process should stop: [`stop`](Self::stop) or [`fail`](Self::fail) was called.
    pub fn should_stop(&self) -> bool {
        self.state.load(Ordering::Relaxed)
    }

    /// The stop, if the process was stopped without an error.
    pub fn interruption(&self) -> Option<Interruption> {
        (self.should_stop() && self.error().is_none()).then(|| Interruption {
            reason: self.get_message(),
        })
    }

    /// The error the process stopped on, if [`fail`](Self::fail) was called.
    pub fn error(&self) -> Option<Arc<TrainingError>> {
        self.error.lock().unwrap().clone()
    }

    /// Get the message associated with the interrupt.
    pub fn get_message(&self) -> Option<String> {
        let message = self.message.lock().unwrap();
        message.clone()
    }
}
