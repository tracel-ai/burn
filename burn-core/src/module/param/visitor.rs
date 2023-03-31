use core::marker::PhantomData;

use alloc::vec::Vec;

use super::ParamId;
use crate::module::{Module, ModuleVisitor};
use burn_tensor::{backend::Backend, Tensor};

#[derive(new)]
struct ParamIdCollector<'a, M> {
    param_ids: &'a mut Vec<ParamId>,
    phantom: PhantomData<M>,
}

impl<'a, B, M> ModuleVisitor<B> for ParamIdCollector<'a, M>
where
    B: Backend,
    M: Module<B>,
{
    fn visit<const D: usize>(&mut self, id: &ParamId, _tensor: &Tensor<B, D>) {
        self.param_ids.push(id.clone());
    }
}

/// List all the parameter ids in a module.
pub fn list_param_ids<M: Module<B>, B: Backend>(module: &M) -> Vec<ParamId> {
    let mut params_ids = Vec::new();
    let mut visitor = ParamIdCollector::<M>::new(&mut params_ids);
    module.visit(&mut visitor);

    params_ids
}
