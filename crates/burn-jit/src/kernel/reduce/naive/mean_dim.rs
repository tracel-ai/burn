use crate::{kernel::reduce::MeanDim, JitElement};
use burn_cube::{
    cpa,
    dialect::{Item, Scope, Variable},
};

use super::base::ReduceDimNaive;

impl<E: JitElement> ReduceDimNaive<E> for MeanDim {
    type Accumulator = Variable;

    fn initialize_naive(scope: &mut Scope, _input_item: Item, output_item: Item) -> Variable {
        scope.zero(output_item)
    }

    fn inner_loop_naive(scope: &mut Scope, accumulator: Variable, value: Variable, _i: Variable) {
        cpa!(scope, accumulator += value);
    }

    fn assign_naive(
        scope: &mut Scope,
        output: Variable,
        accumulator: Variable,
        shape_reduce_dim: Variable,
    ) {
        let id = Variable::Id;
        let denominator = scope.create_local(accumulator.item());
        cpa!(scope, denominator = cast(shape_reduce_dim));
        cpa!(scope, accumulator = accumulator / denominator);
        cpa!(scope, output[id] = accumulator);
    }
}
