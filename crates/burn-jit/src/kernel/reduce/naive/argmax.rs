use crate::{kernel::reduce::Argmax, JitElement};
use cubecl::{
    cpa,
    ir::{Elem, Item, Scope, Variable},
};

use super::base::ReduceDimNaive;

impl<E: JitElement> ReduceDimNaive<E> for Argmax {
    type Accumulator = (Variable, Variable);

    fn initialize_naive(
        scope: &mut Scope,
        input_item: Item,
        _output_item: Item,
    ) -> Self::Accumulator {
        let index = scope.create_local(Elem::UInt);
        let max = scope.create_local(input_item);
        let max_initial = input_item
            .elem()
            .constant_from_f64(E::minimum_value().to_f64());
        cpa!(scope, max = max_initial);

        (max, index)
    }

    fn inner_loop_naive(
        scope: &mut Scope,
        (max, index): Self::Accumulator,
        value: Variable,
        i: Variable,
    ) {
        let condition = scope.create_local(Elem::Bool);
        cpa!(scope, condition = value > max);
        cpa!(scope, if(condition).then(|scope| {
            cpa!(scope, max = value);
            cpa!(scope, index = i);
        }));
    }

    fn assign_naive(
        scope: &mut Scope,
        output: Variable,
        (_max, index): Self::Accumulator,
        _shape_reduce_dim: Variable,
    ) {
        let id = Variable::AbsolutePos;
        cpa!(scope, output[id] = index);
    }
}
