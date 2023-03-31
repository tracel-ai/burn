use core::marker::PhantomData;

use super::{GradientsParams, Optimizer};
use crate::module::{ADModule, ModuleVisitor, ParamId, StateNamed};
use burn_tensor::{backend::ADBackend, Tensor};

#[derive(new)]
pub struct GradientsRegister<'a, M: ADModule<B>, B: ADBackend, O> {
    optimizer: &'a O,
    state: &'a mut StateNamed<B::FloatElem>,
    phatom: PhantomData<M>,
}

#[derive(new)]
pub struct GradientsLoader<'a, M: ADModule<B>, B: ADBackend, O> {
    optimizer: &'a mut O,
    state: &'a StateNamed<B::FloatElem>,
    phatom: PhantomData<M>,
}

#[derive(new)]
pub struct GradientsParamsConverter<'a, M: ADModule<B>, B: ADBackend> {
    grads: B::Gradients,
    grads_params: &'a mut GradientsParams,
    phatom: PhantomData<M>,
}

#[derive(new)]
pub struct GradientsParamsChangeDevice<'a, M: ADModule<B>, B: ADBackend> {
    device: &'a B::Device,
    grads: &'a mut GradientsParams,
    phatom: PhantomData<M>,
}

impl<'a, B, M, O> ModuleVisitor<B> for GradientsRegister<'a, M, B, O>
where
    B: ADBackend,
    M: ADModule<B>,
    O: Optimizer<M, B>,
{
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        self.optimizer.register_param_state::<D>(id, self.state)
    }
}

impl<'a, B, M, O> ModuleVisitor<B> for GradientsLoader<'a, M, B, O>
where
    B: ADBackend,
    M: ADModule<B>,
    O: Optimizer<M, B>,
{
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        self.optimizer
            .load_param_state::<D>(id, self.state, &tensor.device())
    }
}

impl<'a, B, M> ModuleVisitor<B> for GradientsParamsConverter<'a, M, B>
where
    B: ADBackend,
    M: ADModule<B>,
{
    fn visit<const D: usize>(&mut self, id: &ParamId, tensor: &Tensor<B, D>) {
        if let Some(grad) = tensor.grad_remove(&mut self.grads) {
            self.grads_params
                .register::<B::InnerBackend, D>(id.clone(), grad);
        }
    }
}

impl<'a, B, M> ModuleVisitor<B> for GradientsParamsChangeDevice<'a, M, B>
where
    B: ADBackend,
    M: ADModule<B>,
{
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) {
            self.grads
                .register::<B::InnerBackend, D>(id.clone(), grad.to_device(self.device));
        }
    }
}
