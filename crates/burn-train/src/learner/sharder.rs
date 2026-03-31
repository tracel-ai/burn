use burn_core::{
    Tensor,
    module::{Module, ModuleMapper, Param},
    tensor::backend::{AutodiffBackend, distributed::DistributedParamId},
};

use crate::{Learner, LearningComponentsTypes};

/// Describes how the module is distributed across multiple devices.
pub struct ModuleSharder;

impl<B: AutodiffBackend> ModuleMapper<B> for ModuleSharder {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = tensor.set_distributed(DistributedParamId::from(id.val()));
        Param::from_mapped_value(id, tensor, mapper)
    }
}

impl<LC: LearningComponentsTypes> Learner<LC> {
    /// Mark the model as sharded across multiple devices.
    pub fn grad_sharded(&mut self) {
        self.model = self.model.clone().map(&mut ModuleSharder);
    }
}
