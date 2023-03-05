use super::{GradientsParams, Optimizer};
use crate::module::{ModuleVisitor, ParamId, StateNamed};
use burn_tensor::{backend::ADBackend, Tensor};

#[derive(new)]
pub struct GradientsRegister<'a, B: ADBackend, O> {
    optimizer: &'a O,
    state: &'a mut StateNamed<B::FloatElem>,
}

#[derive(new)]
pub struct GradientsLoader<'a, B: ADBackend, O> {
    optimizer: &'a mut O,
    state: &'a StateNamed<B::FloatElem>,
}

#[derive(new)]
pub struct GradientsParamsConverter<'a, B: ADBackend> {
    grads: B::Gradients,
    grads_params: &'a mut GradientsParams,
}

#[derive(new)]
pub struct GradientsParamsChangeDevice<'a, B: ADBackend> {
    device: &'a B::Device,
    grads: &'a mut GradientsParams,
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleVisitor<B> for GradientsRegister<'a, B, O> {
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        self.optimizer.register_param_state::<D>(id, self.state)
    }
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleVisitor<B> for GradientsLoader<'a, B, O> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        self.optimizer
            .load_param_state::<D>(id, self.state, &tensor.device())
    }
}

impl<'a, B: ADBackend> ModuleVisitor<B> for GradientsParamsConverter<'a, B> {
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        if let Some(grad) = tensor.grad_remove(&mut self.grads) {
            self.grads_params
                .register::<B::InnerBackend, D>(id.clone(), grad);
        }
    }
}

impl<'a, B: ADBackend> ModuleVisitor<B> for GradientsParamsChangeDevice<'a, B> {
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) {
            self.grads
                .register::<B::InnerBackend, D>(id.clone(), grad.to_device(self.device));
        }
    }
}
