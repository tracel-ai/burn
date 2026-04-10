use burn_core as burn;

use super::GradientsParams;
use burn::module::{AutodiffModule, ModuleVisitor, Param, ParamId};
use burn::tensor::{Device, Gradients, Tensor};
use core::marker::PhantomData;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[derive(new)]
pub struct GradientsParamsConverter<'a, M: AutodiffModule> {
    grads: &'a mut Gradients,
    grads_params: &'a mut GradientsParams,
    phatom: PhantomData<M>,
    filter: Option<Vec<ParamId>>,
}

#[derive(new)]
pub struct GradientsParamsChangeDevice<'a, M: AutodiffModule> {
    device: &'a Device,
    grads: &'a mut GradientsParams,
    phatom: PhantomData<M>,
}

impl<M> ModuleVisitor for GradientsParamsConverter<'_, M>
where
    M: AutodiffModule,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if let Some(filter) = self.filter.as_ref()
            && !filter.contains(&param.id)
        {
            return;
        }

        let Some(grad) = param.val().grad_remove(self.grads) else {
            return;
        };

        self.grads_params.register(param.id, grad);
    }
}

impl<M> ModuleVisitor for GradientsParamsChangeDevice<'_, M>
where
    M: AutodiffModule,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(grad) = self.grads.remove::<D>(param.id) else {
            return;
        };

        self.grads
            .register::<D>(param.id, grad.to_device(self.device));
    }
}
