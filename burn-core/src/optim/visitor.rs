use super::GradientsParams;
use crate::module::{ADModule, ModuleVisitor, ParamId};
use burn_tensor::{backend::ADBackend, Tensor};
use core::marker::PhantomData;

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
