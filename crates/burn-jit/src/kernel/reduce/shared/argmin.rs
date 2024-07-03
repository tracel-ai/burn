use burn_cube::{
    cpa,
    ir::{Elem, Item, Scope, Variable},
};

use crate::{kernel::reduce::Argmin, JitElement};

use super::base::ReduceDimShared;

impl<E: JitElement> ReduceDimShared<E> for Argmin {
    type Accumulator = (Variable, Variable);

    fn initialize_shared(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator {
        let value_shared_memory = scope.create_shared(input_item, shared_memory_size);
        let index_shared_memory = scope.create_shared(Elem::UInt, shared_memory_size);

        let min = Variable::ConstantScalar(E::maximum_value().to_f64(), input_item.elem());
        cpa!(scope, value_shared_memory[write_position] = min);
        (value_shared_memory, index_shared_memory)
    }

    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        (value, index): Self::Accumulator,
    ) {
        let (value_shared_memory, index_shared_memory) = shared_memory;
        let current_value = scope.create_local(value.item());
        cpa!(scope, current_value = value_shared_memory[write_position]);

        let condition = scope.create_local(Elem::Bool);
        cpa!(scope, condition = value < current_value);
        cpa!(scope, if(condition).then(|scope| {
            cpa!(scope, value_shared_memory[write_position] = value);
            cpa!(scope, index_shared_memory[write_position] = index);
        }));
    }

    fn read_from_input(
        scope: &mut Scope,
        input: Variable,
        read_position: Variable,
        i: Variable,
    ) -> Self::Accumulator {
        let value = scope.create_local(input.item());
        cpa!(scope, value = input[read_position]);
        (value, i)
    }

    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Self::Accumulator {
        let (value_shared_memory, index_shared_memory) = shared_memory;
        let value = scope.create_local(value_shared_memory.item());
        cpa!(scope, value = value_shared_memory[read_position]);
        let index = scope.create_local(index_shared_memory.item());
        cpa!(scope, index = index_shared_memory[read_position]);
        (value, index)
    }

    fn assign_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        output: Variable,
        write_position: Variable,
        _shape_reduce_dim: Variable,
    ) {
        let (_, index_shared_memory) = shared_memory;
        let final_value = scope.create_local(output.item());
        cpa!(scope, final_value = index_shared_memory[0]);
        cpa!(scope, output[write_position] = final_value);
    }
}
