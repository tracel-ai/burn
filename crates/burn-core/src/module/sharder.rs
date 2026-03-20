use burn_tensor::{
    Tensor,
    backend::Backend,
    communication::{DistributedParamId, PeerId, ReduceOperation},
};

use crate::module::{ModuleMapper, Param};

/// Describes how the module is distributed across multiple devices.
pub struct ModuleSharder {
    /// The device's [PeerId].
    pub peer_id: PeerId,
    /// The reduce operation.
    pub op: ReduceOperation,
}

impl<B: Backend> ModuleMapper<B> for ModuleSharder {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = tensor.set_distributed_params(
            self.peer_id,
            self.op,
            DistributedParamId::from(id.val()),
        );
        Param::from_mapped_value(id, tensor, mapper)
    }
}
