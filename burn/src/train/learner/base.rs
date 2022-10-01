use crate::module::ADModule;
use crate::optim::Optimizer;
use crate::tensor::backend::Backend;
use crate::train::checkpoint::Checkpointer;
use crate::train::LearnerCallback;

#[derive(new)]
pub struct Learner<M, O, CM, CO, TO, VO> {
    pub(super) model: M,
    pub(super) optim: O,
    pub(super) checkpointer_model: Option<CM>,
    pub(super) checkpointer_optimizer: Option<CO>,
    pub(super) callback: Box<dyn LearnerCallback<TO, VO>>,
    pub(super) num_epochs: usize,
    pub(super) checkpoint: Option<usize>,
}

impl<M, O, CM, CO, TO, VO> Learner<M, O, CM, CO, TO, VO>
where
    VO: Send + Sync + 'static,
    TO: Send + Sync + 'static,
    M: ADModule,
    O: Optimizer<Backend = M::Backend>,
    CM: Checkpointer<<M::Backend as Backend>::Elem>,
    CO: Checkpointer<<M::Backend as Backend>::Elem>,
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
