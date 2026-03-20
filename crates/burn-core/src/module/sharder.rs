use burn_tensor::{Tensor, backend::Backend, communication::DistributedParamId};

use crate::module::{ModuleMapper, Param};

/// Describes how the module is distributed across multiple devices.
pub struct ModuleSharder;

impl<B: Backend> ModuleMapper<B> for ModuleSharder {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = tensor.set_distributed_params(DistributedParamId::from(id.val()));
        Param::from_mapped_value(id, tensor, mapper)
    }
}
