use super::{Param, ParamId};
use crate::module::{Module, ModuleVisitor};
use alloc::vec::Vec;
use burn_tensor::{Bool, Int, Tensor};
use core::marker::PhantomData;

struct ParamIdCollector<'a, M> {
    param_ids: &'a mut Vec<ParamId>,
    phantom: PhantomData<M>,
}

impl<M> ModuleVisitor for ParamIdCollector<'_, M>
where
    M: Module,
{
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.param_ids.push(param.id);
    }
    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        self.param_ids.push(param.id);
    }
    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
        self.param_ids.push(param.id);
    }
}

/// Lists the IDs of all tensor-backed parameters in a module.
///
/// This includes floating-point, integer and boolean tensor parameters. Module-owned non-tensor
/// control values such as [`Flag`](crate::module::Flag) are excluded.
///
/// To create a group containing every module-owned value in a subtree, including flags, use
/// [`ParamGroup::ids_from_module`](crate::module::ParamGroup::ids_from_module).
pub fn list_param_ids<M: Module>(module: &M) -> Vec<ParamId> {
    let mut params_ids = Vec::new();
    let mut visitor = ParamIdCollector {
        param_ids: &mut params_ids,
        phantom: PhantomData::<M>,
    };
    module.visit(&mut visitor);

    params_ids
}
