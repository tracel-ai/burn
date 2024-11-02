use crate::{kernel::reduce::Argmax, JitElement};
use burn_tensor::cast::ToElement;
use cubecl::{
    cpa,
    ir::{Elem, Item, Scope, Variable},
    prelude::*,
};

use super::base::{ReduceDimShared, ReduceDimSharedCube};

impl<E: JitElement> ReduceDimShared<E> for Argmax {
    type Accumulator = (Variable, Variable);

    fn initialize_shared(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator {
        let value_shared_memory = scope.create_shared(input_item, shared_memory_size);
        let index_shared_memory = scope.create_shared(u32::as_elem(), shared_memory_size);
        let max = input_item
            .elem()
            .constant_from_f64(ToElement::to_f64(&E::minimum_value()));
        cpa!(scope, value_shared_memory[write_position] = max);
        (value_shared_memory, index_shared_memory)
    }

    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        (value, index): Self::Accumulator,
    ) {
        let (value_shared_memory, index_shared_memory) = shared_memory;
        let current_value = scope.create_local(value.item);
        cpa!(scope, current_value = value_shared_memory[write_position]);

        let condition = scope.create_local(Elem::Bool);
        cpa!(scope, condition = value > current_value);
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
        let value = scope.create_local(input.item);
        cpa!(scope, value = input[read_position]);
        (value, i)
    }

    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Self::Accumulator {
        let (value_shared_memory, index_shared_memory) = shared_memory;
        let value = scope.create_local(value_shared_memory.item);
        cpa!(scope, value = value_shared_memory[read_position]);
        let index = scope.create_local(index_shared_memory.item);
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
        let final_value = scope.create_local(index_shared_memory.item);
        cpa!(scope, final_value = index_shared_memory[0]);
        cpa!(scope, output[write_position] = final_value);
    }
}

#[cube]
impl<EIn: JitElement, EOut: JitElement> ReduceDimSharedCube<EIn, EOut> for Argmax {
    /// The reduction accumulator
    type Accumulator = (SharedMemory<EIn>, SharedMemory<u32>);
    type Value = (EIn, u32);

    /// Initialization for shared algorithm
    fn initialize_shared(
        shared_memory_size: u32,
        write_position: u32,
    ) -> (SharedMemory<EIn>, SharedMemory<u32>) {
        let mut value_shared = SharedMemory::new(shared_memory_size);
        let mut index_shared = SharedMemory::new(shared_memory_size);
        value_shared[write_position] = comptime![EIn::minimum_value()].runtime();
        index_shared[write_position] = 0;
        (value_shared, index_shared)
    }

    /// How to write to shared memory
    fn write_to_shared(
        shared_memory: &mut (SharedMemory<EIn>, SharedMemory<u32>),
        write_position: u32,
        value: (EIn, u32),
    ) {
        let (values, indices) = shared_memory;
        let (value, index) = value;

        if values[write_position] < value {
            values[write_position] = value;
            indices[write_position] = index;
        }
    }

    /// How to read from input in shared algorithm
    fn read_from_input(input: &Tensor<EIn>, read_position: u32, i: u32) -> (EIn, u32) {
        (input[read_position], i)
    }

    /// How to read from shared memory
    fn read_from_shared(
        shared_memory: &(SharedMemory<EIn>, SharedMemory<u32>),
        read_position: u32,
    ) -> (EIn, u32) {
        let (values, indices) = shared_memory;
        (values[read_position], indices[read_position])
    }

    /// How to assign from shared memory
    fn assign_shared(
        shared_memory: &(SharedMemory<EIn>, SharedMemory<u32>),
        output: &mut Tensor<EOut>,
        write_position: u32,
        _shape_reduce_dim: u32,
    ) {
        let (_, indices) = shared_memory;
        output[write_position] = EOut::cast_from(indices[0]);
    }
}
