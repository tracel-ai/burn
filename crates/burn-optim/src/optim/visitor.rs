use burn_core as burn;

use super::GradientsParams;
use burn::module::{AutodiffModule, ModuleVisitor, ParamId};
use burn::tensor::{Tensor, backend::AutodiffBackend};
use core::marker::PhantomData;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

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

impl<B, M> ModuleVisitor<B> for GradientsParamsConverter<'_, M, B>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    fn visit_float<const D: usize>(&mut self, id: ParamId, tensor: &Tensor<B, D>) {
        if let Some(filter) = self.filter.as_ref()
            && !filter.contains(&id)
        {
            return;
        }

        let Some(grad) = tensor.grad_remove(self.grads) else {
            return;
        };

        self.grads_params.register::<B::InnerBackend, D>(id, grad);
    }
}

impl<B, M> ModuleVisitor<B> for GradientsParamsChangeDevice<'_, M, B>
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
