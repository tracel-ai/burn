use alloc::vec::Vec;

use super::ParamId;
use crate::module::{Module, ModuleVisitor};
use burn_tensor::{backend::Backend, Tensor};

#[derive(new)]
struct ParamIdCollector<'a> {
    param_ids: &'a mut Vec<ParamId>,
}

impl<'a, B: Backend> ModuleVisitor<B> for ParamIdCollector<'a> {
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        self.param_ids.push(id.clone());
    }
}

/// List all the parameter ids in a module.
pub fn list_param_ids<M: Module>(module: &M) -> Vec<ParamId> {
    let mut params_ids = Vec::new();
    let mut visitor = ParamIdCollector::new(&mut params_ids);
    module.visit(&mut visitor);

    params_ids
}
