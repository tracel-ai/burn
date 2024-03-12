use crate::{
    codegen::dialect::gpu::{gpu, Item, Scope, Variable},
    JitElement,
};

use super::ReduceDimAlgorithm;

pub(crate) struct MeanDim;

impl<E: JitElement> ReduceDimAlgorithm<E> for MeanDim {
    type Accumulator = Variable;

    fn initialize_naive(scope: &mut Scope, _input_item: Item, output_item: Item) -> Variable {
        scope.zero(output_item)
    }

    fn inner_loop_naive(scope: &mut Scope, accumulator: Variable, value: Variable, _i: Variable) {
        gpu!(scope, accumulator += value);
    }

    fn assign_naive(
        scope: &mut Scope,
        output: Variable,
        accumulator: Variable,
        shape_reduce_dim: Variable,
    ) {
        let id = Variable::Id;
        let denominator = scope.create_local(accumulator.item());
        gpu!(scope, denominator = cast(shape_reduce_dim));
        gpu!(scope, accumulator = accumulator / denominator);
        gpu!(scope, output[id] = accumulator);
    }

    fn initialize_shared(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator {
        let shared_memory = scope.create_shared(input_item, shared_memory_size);
        let neutral_element = scope.zero(shared_memory.item());
        gpu!(scope, shared_memory[write_position] = neutral_element);
        shared_memory
    }

    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        value: Self::Accumulator,
    ) {
        let current_value = scope.create_local(value.item());
        let computed = scope.create_local(value.item());
        gpu!(scope, current_value = shared_memory[write_position]);
        gpu!(scope, computed = current_value + value);
        gpu!(scope, shared_memory[write_position] = computed);
    }

    fn read_from_input(
        scope: &mut Scope,
        input: Variable,
        read_position: Variable,
        _i: Variable,
    ) -> Self::Accumulator {
        let value = scope.create_local(input.item());
        gpu!(scope, value = input[read_position]);
        value
    }

    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Variable {
        let read_value = scope.create_local(shared_memory.item());
        gpu!(scope, read_value = shared_memory[read_position]);
        read_value
    }

    fn assign_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        output: Variable,
        write_position: Variable,
        shape_reduce_dim: Variable,
    ) {
        let final_value = scope.create_local(output.item());
        gpu!(scope, final_value = shared_memory[0]);

        let denominator = scope.create_local(output.item());
        gpu!(scope, denominator = cast(shape_reduce_dim));
        gpu!(scope, final_value = final_value / denominator);
        gpu!(scope, output[write_position] = final_value);
    }
}
