use crate::checkpoint::Checkpointer;
use crate::LearnerCallback;
use burn_core::module::{ADModule, Module};
use burn_core::optim::Optimizer;
use burn_core::tensor::backend::Backend;

/// Learner struct encapsulating all components necessary to train a Neural Network model.
///
/// To create a learner, use the [builder](crate::train::LearnerBuilder) struct.
pub struct Learner<M, O, TO, VO>
where
    M: ADModule,
{
    pub(super) model: M,
    pub(super) optim: O,
    pub(super) num_epochs: usize,
    pub(super) callback: Box<dyn LearnerCallback<TO, VO>>,
    pub(super) checkpoint: Option<usize>,
    pub(super) checkpointer_model: CheckpointModel<M>,
    pub(super) checkpointer_optimizer: CheckpointOptim<M>,
    pub(super) grad_accumulation: Option<usize>,
    pub(super) devices: Vec<<M::Backend as Backend>::Device>,
}

type CheckpointModel<M> =
    Option<Box<dyn Checkpointer<<<M as Module>::Backend as Backend>::FloatElem>>>;
type CheckpointOptim<M> =
    Option<Box<dyn Checkpointer<<<M as Module>::Backend as Backend>::FloatElem>>>;

impl<M, O, TO, VO> Learner<M, O, TO, VO>
where
    VO: Send + Sync + 'static,
    TO: Send + Sync + 'static,
    M: ADModule,
    O: Optimizer<Backend = M::Backend>,
{
    pub(super) fn checkpoint(
        model: &M,
        optim: &O,
        checkpointer_model: &CheckpointModel<M>,
        checkpointer_optimizer: &CheckpointOptim<M>,
        epoch: usize,
    ) {
        if let Some(checkpointer) = &checkpointer_model {
            checkpointer.save(epoch, model.state()).unwrap();
        }
        if let Some(checkpointer) = &checkpointer_optimizer {
            checkpointer.save(epoch, optim.state(model)).unwrap();
        }
    }

    pub(super) fn load_checkpoint(mut self, epoch: usize) -> Self {
        if let Some(checkpointer) = &self.checkpointer_model {
            let state = checkpointer.restore(epoch).unwrap();
            self.model = self.model.load(&state).unwrap();
        }

        if let Some(checkpointer) = &self.checkpointer_optimizer {
            let state = checkpointer.restore(epoch).unwrap();
            self.optim.load(&self.model, &state).unwrap();
        }

        self
    }
}
