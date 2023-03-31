use core::marker::PhantomData;

use burn_tensor::{backend::ADBackend, Tensor};

use crate::module::{ADModule, ModuleMapper, ParamId};

use super::{GradientsParams, Optimizer};

#[derive(new)]
pub struct ModuleTensorUpdater<'a, M, O> {
    optimizer: &'a mut O,
    grads: GradientsParams,
    phatom: PhantomData<M>,
}

impl<'a, M, B, O> ModuleMapper<B> for ModuleTensorUpdater<'a, M, O>
where
    M: ADModule<B>,
    B: ADBackend,
    O: Optimizer<M, B>,
{
    fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) {
            self.optimizer.update_tensor(id, tensor, grad)
        } else {
            tensor
        }
    }
}
