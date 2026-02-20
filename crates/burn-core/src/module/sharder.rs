use burn_tensor::{
    Tensor,
    backend::{Backend, ModuleParamId, PeerId, ReduceOperation},
};

use crate::module::{ModuleMapper, Param};

/// Describes how shard a module across multiple devices for DDP.
pub struct ModuleSharder {
    /// The calibration method used in quantization.
    pub peer_id: PeerId,
    /// The quantization scheme.
    pub op: ReduceOperation,
}

impl<B: Backend> ModuleMapper<B> for ModuleSharder {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor =
            tensor.set_sharded_params(self.peer_id, self.op, Some(ModuleParamId::from(id.val())));
        Param::from_mapped_value(id, tensor, mapper)
    }
}
