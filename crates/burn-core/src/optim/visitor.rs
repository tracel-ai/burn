use super::GradientsParams;
use crate::module::{AutodiffModule, ModuleVisitor, ParamId};
use burn_tensor::{backend::AutodiffBackend, Tensor};
use core::marker::PhantomData;

#[derive(new)]
pub struct GradientsParamsConverter<'a, M: AutodiffModule<B>, B: AutodiffBackend> {
    grads: &'a mut B::Gradients,
    grads_params: &'a mut GradientsParams,
    phatom: PhantomData<M>,
    filter: Option<Vec<ParamId>>,
}

#[derive(new)]
pub struct GradientsParamsChangeDevice<'a, M: AutodiffModule<B>, B: AutodiffBackend> {
    device: &'a B::Device,
    grads: &'a mut GradientsParams,
    phatom: PhantomData<M>,
}

impl<'a, B, M> ModuleVisitor<B> for GradientsParamsConverter<'a, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
        if let Some(filter) = self.filter.as_ref() {
            if !filter.contains(&id) {
                return;
            }
        }
        let Some(grad) = tensor.grad_remove(self.grads) else {
            return;
        };

        self.grads_params.register::<B::InnerBackend, D>(id, grad);
    }
}

impl<'a, B, M> ModuleVisitor<B> for GradientsParamsChangeDevice<'a, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
        let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) else {
            return;
        };

        self.grads
            .register::<B::InnerBackend, D>(id, grad.to_device(self.device));
    }
}
