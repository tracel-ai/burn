use crate::checkpoint::Checkpointer;
use crate::LearnerCallback;
use burn_core::lr_scheduler::LearningRateScheduler;
use burn_core::module::{ADModule, Module};
use burn_core::optim::Optimizer;
use burn_core::tensor::backend::ADBackend;

/// Learner struct encapsulating all components necessary to train a Neural Network model.
///
/// To create a learner, use the [builder](crate::train::LearnerBuilder) struct.
pub struct Learner<B, M, O, S, TO, VO>
where
    B: ADBackend,
    M: ADModule<B>,
    O: Optimizer<M, B>,
    S: LearningRateScheduler,
{
    pub(super) model: M,
    pub(super) optim: O,
    pub(super) scheduler: S,
    pub(super) num_epochs: usize,
    pub(super) callback: Box<dyn LearnerCallback<TO, VO>>,
    pub(super) checkpoint: Option<usize>,
    pub(super) checkpointer_model: CheckpointModel<M, B>,
    pub(super) checkpointer_optimizer: CheckpointOptim<O, M, B>,
    pub(super) checkpointer_scheduler: CheckpointScheduler<S>,
    pub(super) grad_accumulation: Option<usize>,
    pub(super) devices: Vec<B::Device>,
}

type CheckpointModel<M, B> = Option<Box<dyn Checkpointer<<M as Module<B>>::Record>>>;
type CheckpointOptim<O, M, B> = Option<Box<dyn Checkpointer<<O as Optimizer<M, B>>::Record>>>;
type CheckpointScheduler<S> = Option<Box<dyn Checkpointer<<S as LearningRateScheduler>::Record>>>;

impl<B, M, O, S, TO, VO> Learner<B, M, O, S, TO, VO>
where
    VO: Send + Sync + 'static,
    TO: Send + Sync + 'static,
    B: ADBackend,
    M: ADModule<B>,
    O: Optimizer<M, B>,
    S: LearningRateScheduler,
{
    pub(super) fn checkpoint(
        model: &M,
        optim: &O,
        scheduler: &S,
        checkpointer_model: &CheckpointModel<M, B>,
        checkpointer_optimizer: &CheckpointOptim<O, M, B>,
        checkpointer_scheduler: &CheckpointScheduler<S>,
        epoch: usize,
    ) {
        if let Some(checkpointer) = &checkpointer_model {
            checkpointer
                .save(epoch, model.clone().into_record())
                .unwrap();
        }

        if let Some(checkpointer) = &checkpointer_optimizer {
            checkpointer.save(epoch, optim.to_record()).unwrap();
        }

        if let Some(checkpointer) = &checkpointer_scheduler {
            checkpointer.save(epoch, scheduler.to_record()).unwrap();
        }
    }

    pub(super) fn load_checkpoint(mut self, epoch: usize) -> Self {
        if let Some(checkpointer) = &self.checkpointer_model {
            let record = checkpointer.restore(epoch).unwrap();
            self.model = self.model.load_record(record);
        }

        if let Some(checkpointer) = &self.checkpointer_optimizer {
            let record = checkpointer.restore(epoch).unwrap();
            self.optim = self.optim.load_record(record);
        }

        if let Some(checkpointer) = &self.checkpointer_scheduler {
            let record = checkpointer.restore(epoch).unwrap();
            self.scheduler = self.scheduler.load_record(record);
        }

        self
    }
}
