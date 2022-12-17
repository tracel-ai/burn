use crate::module::ADModule;
use crate::optim::Optimizer;
use crate::tensor::backend::Backend;
use crate::train::checkpoint::Checkpointer;
use crate::train::LearnerCallback;

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
    pub(super) checkpointer_model: Option<Box<dyn Checkpointer<<M::Backend as Backend>::Elem>>>,
    pub(super) checkpointer_optimizer: Option<Box<dyn Checkpointer<<M::Backend as Backend>::Elem>>>,
}

impl<M, O, TO, VO> Learner<M, O, TO, VO>
where
    VO: Send + Sync + 'static,
    TO: Send + Sync + 'static,
    M: ADModule,
    O: Optimizer<Backend = M::Backend>,
{
    pub(super) fn checkpoint(&self, epoch: usize) {
        if let Some(checkpointer) = &self.checkpointer_model {
            checkpointer.save(epoch, self.model.state()).unwrap();
        }
        if let Some(checkpointer) = &self.checkpointer_optimizer {
            checkpointer
                .save(epoch, self.optim.state(&self.model))
                .unwrap();
        }
    }

    pub(super) fn load_checkpoint(&mut self, epoch: usize) {
        if let Some(checkpointer) = &self.checkpointer_model {
            let state = checkpointer.restore(epoch).unwrap();
            self.model.load(&state).unwrap();
        }

        if let Some(checkpointer) = &self.checkpointer_optimizer {
            let state = checkpointer.restore(epoch).unwrap();
            self.optim.load(&self.model, &state).unwrap();
        }
    }
}
