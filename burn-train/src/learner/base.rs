use crate::checkpoint::Checkpointer;
use crate::components::TrainingComponents;
use burn_core::lr_scheduler::LrScheduler;
use burn_core::module::Module;
use burn_core::optim::Optimizer;
use burn_core::tensor::backend::Backend;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Learner struct encapsulating all components necessary to train a Neural Network model.
///
/// To create a learner, use the [builder](crate::learner::LearnerBuilder) struct.
pub struct Learner<T: TrainingComponents> {
    pub(crate) model: T::Model,
    pub(crate) optim: T::Optimizer,
    pub(crate) lr_scheduler: T::LrScheduler,
    pub(crate) num_epochs: usize,
    pub(crate) checkpoint: Option<usize>,
    pub(crate) grad_accumulation: Option<usize>,
    pub(crate) checkpointer: Option<TrainingCheckpointer<T>>,
    pub(crate) devices: Vec<<T::Backend as Backend>::Device>,
    pub(crate) callback: T::Callback,
    pub(crate) interrupter: TrainingInterrupter,
}

#[derive(new)]
pub(crate) struct TrainingCheckpointer<T: TrainingComponents> {
    model: T::CheckpointerModel,
    optim: T::CheckpointerOptimizer,
    lr_scheduler: T::CheckpointerLrScheduler,
}

impl<T: TrainingComponents> TrainingCheckpointer<T> {
    pub(crate) fn checkpoint(
        &self,
        model: &T::Model,
        optim: &T::Optimizer,
        scheduler: &T::LrScheduler,
        epoch: usize,
    ) {
        self.model.save(epoch, model.clone().into_record()).unwrap();
        self.optim.save(epoch, optim.to_record()).unwrap();
        self.lr_scheduler
            .save(epoch, scheduler.to_record())
            .unwrap();
    }

    pub(crate) fn load_checkpoint(
        &self,
        model: T::Model,
        optim: T::Optimizer,
        scheduler: T::LrScheduler,
        epoch: usize,
    ) -> (T::Model, T::Optimizer, T::LrScheduler) {
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
