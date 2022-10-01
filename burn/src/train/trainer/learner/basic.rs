use super::{CheckpointModel, TrainStep, ValidStep};
use crate::module::ADModule;
use crate::optim::Optimizer;
use crate::tensor::backend::Backend;
use crate::train::checkpoint::{AsyncCheckpointer, Checkpointer, FileCheckpointer};
use burn_tensor::{Element, Gradients};
use std::sync::Arc;

#[derive(new)]
pub struct TrainerModel<M, O, CM, CO> {
    pub model: M,
    pub optim: O,
    checkpointer_model: Option<CM>,
    checkpointer_optimizer: Option<CO>,
}

#[derive(Default)]
pub struct TrainerModelBuilder<B: Backend> {
    checkpointer_model: Option<Arc<dyn Checkpointer<B::Elem> + Send + Sync>>,
    checkpointer_optimizer: Option<Arc<dyn Checkpointer<B::Elem> + Send + Sync>>,
}

impl<B: Backend> TrainerModelBuilder<B> {
    pub fn with_file_checkpointer<P: Element + serde::de::DeserializeOwned + serde::Serialize>(
        mut self,
        directory: &str,
    ) -> Self {
        self.checkpointer_model = Some(Arc::new(FileCheckpointer::<P>::new(directory, "model")));
        self.checkpointer_optimizer =
            Some(Arc::new(FileCheckpointer::<P>::new(directory, "optim")));
        self
    }

    pub fn build<M, O>(
        self,
        model: M,
        optim: O,
    ) -> TrainerModel<M, O, AsyncCheckpointer<B::Elem>, AsyncCheckpointer<B::Elem>> {
        TrainerModel {
            model,
            optim,
            checkpointer_model: self.checkpointer_model.map(AsyncCheckpointer::new),
            checkpointer_optimizer: self.checkpointer_optimizer.map(AsyncCheckpointer::new),
        }
    }
}

pub trait Backward {
    fn backward(&self) -> Gradients;
}

impl<M, O, CM, CO, TI, TO> TrainStep for TrainerModel<M, O, CM, CO>
where
    M: TrainStep<Input = TI, Output = TO> + ADModule,
    TO: Backward,
    O: Optimizer<Backend = M::Backend>,
{
    type Input = TI;
    type Output = TO;

    fn step(&mut self, item: Self::Input) -> Self::Output {
        let output = self.model.step(item);
        let mut grads = output.backward();

        self.model.update_params(&mut grads, &mut self.optim);

        output
    }
}

impl<M, O, CM, CO, TI, TO> ValidStep for TrainerModel<M, O, CM, CO>
where
    M: ValidStep<Input = TI, Output = TO>,
{
    type Input = TI;
    type Output = TO;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.model.step(item)
    }
}

impl<M, O, CM, CO> CheckpointModel for TrainerModel<M, O, CM, CO>
where
    M: ADModule,
    O: Optimizer<Backend = M::Backend>,
    CM: Checkpointer<<M::Backend as Backend>::Elem>,
    CO: Checkpointer<<M::Backend as Backend>::Elem>,
{
    fn checkpoint(&self, epoch: usize) {
        if let Some(checkpointer) = &self.checkpointer_model {
            checkpointer.save(epoch, self.model.state()).unwrap();
        }
        if let Some(checkpointer) = &self.checkpointer_optimizer {
            checkpointer
                .save(epoch, self.optim.state(&self.model))
                .unwrap();
        }
    }

    fn load_checkpoint(&mut self, epoch: usize) {
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
