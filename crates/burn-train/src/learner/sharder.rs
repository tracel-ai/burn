use burn_core::{
    Tensor,
    module::{Module, ModuleMapper, Param},
    prelude::Backend,
};

use crate::{Learner, LearningComponentsTypes};

/// Describes how the module is distributed across multiple devices.
pub struct ModuleSharder;

impl<B: Backend> ModuleMapper<B> for ModuleSharder {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = tensor.set_distributed_params(DistributedParamId::from(id.val()));
        Param::from_mapped_value(id, tensor, mapper)
    }
}

impl<LC: LearningComponentsTypes> Learner<LC> {
    /// Mark the model as sharded across multiple devices.
    pub fn grad_sharded(&mut self) {
        self.model = self.model.map(&mut ModuleSharder);
    }
}

// NOTES
// Workflow:
// 1. Before training:
//   - set module as distributed
//     `learner.grad_sharded()` -> `module.grad_distributed()` -> `ModuleSharder::map_float` -> set distributed params
//     Right now, this calls `float_set_distributed_params` just so autodiff can mark the `AutodiffTensor` as distributed
//   - initialize communication server: `B::start_communication_server` -> initializes the backend client
//
// 1. During optimization step:
//    optimizer gets the sharded param property, and if set has to set it back afterwards on the `Tensor::from_inner`
//
// 1. During backward:
//    - each op registers the gradients -> on register check if we collected all required grads, if yes then `B::submit_gradient_sync` ->
//      backend client `submit_gradient_sync` -> `B::all_reduce_in_place` (`all_reduce_inplace_centralized` default or cubecl specialization)
//    - (compute gradients) if any distributed params were registered, we need to sync `B::register_sync_parameters` -> backend client `register_sync_parameters`
//
// 1. At the end of backward:
//    `Gradients::sync_collective`` -> `B::submit_sync_collective` to aggregate the grads -> backend client `submit_sync_collective` ->
//    `B::sync_collective` (only implemented for cubecl backends - calls `ComputeClient::sync_collective`)
//
// 1. After training: close communication server: `B::close_communication_server` -> `client.close()` (does nothing?) + removes the backend client from map
