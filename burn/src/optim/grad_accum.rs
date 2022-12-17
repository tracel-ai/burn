use super::{ModuleOptimizer, Optimizer};
use crate::module::{Module, ParamId, StateNamed};
use burn_tensor::{
    backend::{ADBackend, Gradients},
    Tensor,
};

pub struct GradientsAccumulation<O, B: ADBackend> {
    grads: B::Gradients,
    optimizer: O,
    accumulation: usize,
    accumulation_current: usize,
}

impl<B: ADBackend, O> GradientsAccumulation<O, B>
where
    O: Optimizer<Backend = B>,
{
    pub fn new(optim: O, accumulation: usize) -> Self {
        Self {
            grads: B::Gradients::empty(),
            optimizer: optim,
            accumulation,
            accumulation_current: 0,
        }
    }
}

impl<B: ADBackend, O> Optimizer for GradientsAccumulation<O, B>
where
    O: Optimizer<Backend = B>,
{
    type Backend = B;

    fn update_tensor<const D: usize>(
        &mut self,
        id: &ParamId,
        tensor: &mut Tensor<B, D>,
        grads: &B::Gradients,
    ) {
        if self.accumulation_current >= self.accumulation {
            self.optimizer.update_tensor(id, tensor, grads);
            return;
        }

        let id_str = id.to_string();

        let grad = match grads.get::<D>(&id_str) {
            Some(grad) => match self.grads.get::<D>(&id_str) {
                Some(grad_last_step) => grad_last_step.clone() + grad.clone(),
                None => grad.clone(),
            },
            None => match self.grads.get::<D>(&id_str) {
                Some(grad_last_step) => grad_last_step.clone(),
                None => return,
            },
        };

        self.grads.register(id_str, grad);
    }

    fn update_module<M>(&mut self, module: &mut M, grads: &<Self::Backend as ADBackend>::Gradients)
    where
        M: Module<Backend = Self::Backend>,
        Self: Sized,
    {
        self.accumulation_current += 1;

        let mut visitor = ModuleOptimizer::new(self, grads);
        module.visit_mut(&mut visitor);

        if self.accumulation_current >= self.accumulation {
            self.accumulation_current = 0;
        }
    }

    fn register_param_state<const D: usize>(&self, id: &ParamId, state: &mut StateNamed<B::Elem>) {
        self.optimizer.register_param_state::<D>(id, state);
    }

    fn load_param_state<const D: usize>(
        &mut self,
        id: &ParamId,
        state: &StateNamed<B::Elem>,
        device: &B::Device,
    ) {
        self.optimizer.load_param_state::<D>(id, state, device);
    }
}
