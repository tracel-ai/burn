use super::{Param, ParamId};
use crate::module::{Module, ModuleVisitor};
use alloc::vec::Vec;
use burn_tensor::{Bool, Int, Tensor, backend::Backend};
use core::marker::PhantomData;

struct ParamIdCollector<'a, M> {
    param_ids: &'a mut Vec<ParamId>,
    phantom: PhantomData<M>,
}

impl<B, M> ModuleVisitor<B> for ParamIdCollector<'_, M>
where
    B: Backend,
    M: Module<B>,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        self.param_ids.push(param.id);
    }
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<B, D, Int>>) {
        self.param_ids.push(param.id);
    }
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<B, D, Bool>>) {
        self.param_ids.push(param.id);
    }
}

/// List all the parameter ids in a module.
pub fn list_param_ids<M: Module<B>, B: Backend>(module: &M) -> Vec<ParamId> {
    let mut params_ids = Vec::new();
    let mut visitor = ParamIdCollector {
        param_ids: &mut params_ids,
        phantom: PhantomData::<M>,
    };
    module.visit(&mut visitor);

    params_ids
}
