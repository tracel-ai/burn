use super::{Learner, LearnerCheckpoint, Loss};
use crate::module::ADModule;
use crate::optim::Optimizer;
use crate::tensor::backend::{ADBackend, Backend};
use crate::train::checkpoint::{AsyncCheckpointer, Checkpointer, FileCheckpointer};
use crate::train::metric;
use burn_tensor::{Element, Tensor};
use std::sync::Arc;

#[derive(new)]
pub struct BasicLearner<M, O, CM, CO> {
    pub model: M,
    pub optim: O,
    checkpointer_model: Option<CM>,
    checkpointer_optimizer: Option<CO>,
}

#[derive(Default)]
pub struct LearnerBuilder<B: Backend> {
    checkpointer_model: Option<Arc<dyn Checkpointer<B::Elem> + Send + Sync>>,
    checkpointer_optimizer: Option<Arc<dyn Checkpointer<B::Elem> + Send + Sync>>,
}

impl<B: Backend> LearnerBuilder<B> {
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
    ) -> BasicLearner<M, O, AsyncCheckpointer<B::Elem>, AsyncCheckpointer<B::Elem>> {
        BasicLearner {
            model,
            optim,
            checkpointer_model: self.checkpointer_model.map(AsyncCheckpointer::new),
            checkpointer_optimizer: self.checkpointer_optimizer.map(AsyncCheckpointer::new),
        }
    }
}

#[derive(new)]
pub struct BasicOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> metric::Metric<BasicOutput<B>> for metric::LossMetric {
    fn update(&mut self, item: &BasicOutput<B>) -> metric::MetricStateDyn {
        self.update(&item.loss)
    }
    fn clear(&mut self) {
        <metric::LossMetric as metric::Metric<Tensor<B, 1>>>::clear(self);
    }
}

impl<B, B2, T, L, L2, O, CO, CM> Learner<T, T, BasicOutput<B>, BasicOutput<B2>>
    for BasicLearner<L, O, CO, CM>
where
    B: ADBackend<InnerBackend = B2>,
    B2: Backend,
    O: Optimizer<Backend = B>,
    L: Loss<Backend = B, Item = T> + ADModule<Backend = B, InnerModule = L2>,
    L2: Loss<Backend = B::InnerBackend, Item = T>,
{
    type Backend = B;

    fn train(&mut self, item: T) -> BasicOutput<B> {
        let loss = self.model.loss(item);
        let grads = loss.backward();

        self.model.update_params(&grads, &mut self.optim);

        BasicOutput::new(loss)
    }

    fn valid(&self, item: T) -> BasicOutput<B2> {
        let loss = self.model.inner().loss(item);
        BasicOutput::new(loss)
    }
}

impl<M, O, CM, CO> LearnerCheckpoint for BasicLearner<M, O, CM, CO>
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
