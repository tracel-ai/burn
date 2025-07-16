use super::GradientsParams;
use crate::module::{AutodiffModule, ModuleVisitor, ParamId};
use burn_tensor::{Tensor, backend::AutodiffBackend};
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

#[cfg(feature = "collective")]
pub mod collective_gradient_ops {
    use super::*;

    use crate::{
        collective::{DeviceId, SharedAllReduceParams, all_reduce},
        optim::GradientsParams,
    };

    #[derive(new)]
    pub struct GradientsParamsAllReduce<'a, M: AutodiffModule<B>, B: AutodiffBackend> {
        device_id: DeviceId,
        params: SharedAllReduceParams,
        grads: &'a mut GradientsParams,
        m: PhantomData<M>,
        b: PhantomData<B>,
    }

    impl<B, M> ModuleVisitor<B> for GradientsParamsAllReduce<'_, M, B>
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        fn visit_float<const D: usize>(&mut self, id: ParamId, _tensor: &Tensor<B, D>) {
            let Some(mut grad) = self.grads.remove::<B::InnerBackend, D>(id) else {
                return;
            };

            grad = all_reduce(self.device_id, grad, &self.params).unwrap();

            self.grads.register::<B::InnerBackend, D>(id, grad);
        }
    }
}
