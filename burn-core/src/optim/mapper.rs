use burn_tensor::{backend::ADBackend, Tensor};

use crate::module::{ModuleMapper, ParamId};

use super::{GradientsParams, Optimizer};

#[derive(new)]
pub struct ModuleTensorUpdater<'a, O> {
    optimizer: &'a mut O,
    grads: GradientsParams,
}

impl<'a, B: ADBackend, O: Optimizer<Backend = B>> ModuleMapper<B> for ModuleTensorUpdater<'a, O> {
    fn map<const D: usize>(&mut self, id: &ParamId, tensor: Tensor<B, D>) -> Tensor<B, D> {
        if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) {
            self.optimizer.update_tensor(id, tensor, grad)
        } else {
            tensor
        }
    }
}
