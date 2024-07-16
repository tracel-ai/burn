use cubecl::{
    cpa,
    ir::{Elem, Item, Scope, Variable},
};

use crate::{kernel::reduce::Argmin, JitElement};

use super::base::ReduceDimNaive;

impl<E: JitElement> ReduceDimNaive<E> for Argmin {
    type Accumulator = (Variable, Variable);

    fn initialize_naive(
        scope: &mut Scope,
        input_item: Item,
        _output_item: Item,
    ) -> Self::Accumulator {
        let index = scope.create_local(Elem::UInt);
        let min = scope.create_local(input_item);
        let min_initial = input_item
            .elem()
            .constant_from_f64(E::maximum_value().to_f64());

        cpa!(scope, min = min_initial);

        (min, index)
    }

    fn inner_loop_naive(
        scope: &mut Scope,
        (min, index): Self::Accumulator,
        value: Variable,
        i: Variable,
    ) {
        let condition = scope.create_local(Elem::Bool);
        cpa!(scope, condition = value < min);
        cpa!(scope, if(condition).then(|scope| {
            cpa!(scope, min = value);
            cpa!(scope, index = i);
        }));
    }

    fn assign_naive(
        scope: &mut Scope,
        output: Variable,
        (_min, index): Self::Accumulator,
        _shape_reduce_dim: Variable,
    ) {
        let id = Variable::AbsolutePos;
        cpa!(scope, output[id] = index);
    }
}
