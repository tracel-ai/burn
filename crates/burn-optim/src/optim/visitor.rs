use burn_core as burn;

use super::GradientsParams;
use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::tensor::{Device, FloatDType, Gradients, Tensor};
use core::marker::PhantomData;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[derive(new)]
pub struct GradientsParamsConverter<'a, M: Module> {
    grads: &'a mut Gradients,
    grads_params: &'a mut GradientsParams,
    phatom: PhantomData<M>,
    filter: Option<Vec<ParamId>>,
}

/// Collects the squared L2 norm of every gradient registered for a module's parameters.
#[derive(new)]
pub struct GradientsParamsSquaredNorms<'a, M: Module> {
    grads: &'a GradientsParams,
    squares: &'a mut Vec<Tensor<1>>,
    phatom: PhantomData<M>,
}

/// Multiplies every gradient registered for a module's parameters by a scalar tensor.
#[derive(new)]
pub struct GradientsParamsScale<'a, M: Module> {
    grads: &'a mut GradientsParams,
    scale: &'a Tensor<1>,
    phatom: PhantomData<M>,
}

#[derive(new)]
pub struct GradientsParamsChangeDevice<'a, M: Module> {
    device: &'a Device,
    grads: &'a mut GradientsParams,
    phatom: PhantomData<M>,
}

impl<M> ModuleVisitor for GradientsParamsConverter<'_, M>
where
    M: Module,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if let Some(filter) = self.filter.as_ref()
            && !filter.contains(&param.id)
        {
            return;
        }

        // Prevents `grad_remove` from panicking if one of the tensor is not autodiff.
        if !param.is_require_grad() {
            return;
        }

        let Some(grad) = param.val().grad_remove(self.grads) else {
            return;
        };

        self.grads_params.register(param.id, grad);
    }
}

impl<M> ModuleVisitor for GradientsParamsSquaredNorms<'_, M>
where
    M: Module,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(grad) = self.grads.get::<D>(param.id) else {
            return;
        };
        // In F16 the sum of squares overflows quickly, so accumulate in F32 as `clip_by_norm` does.
        let grad = grad.cast(FloatDType::F32);
        self.squares.push(grad.square().sum());
    }
}

impl<M> ModuleVisitor for GradientsParamsScale<'_, M>
where
    M: Module,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(grad) = self.grads.remove::<D>(param.id) else {
            return;
        };
        let dtype = grad.dtype();
        let scale = self.scale.clone().to_device(&grad.device());
        let scaled = grad
            .cast(FloatDType::F32)
            .mul(scale.unsqueeze())
            .cast(dtype);
        self.grads.register::<D>(param.id, scaled);
    }
}

impl<M> ModuleVisitor for GradientsParamsChangeDevice<'_, M>
where
    M: Module,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(grad) = self.grads.remove::<D>(param.id) else {
            return;
        };

        self.grads
            .register::<D>(param.id, grad.to_device(self.device));
    }
}
