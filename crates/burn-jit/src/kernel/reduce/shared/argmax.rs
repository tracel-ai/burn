use crate::{kernel::reduce::Argmax, JitElement};
use cubecl::prelude::*;

use super::base::ReduceDimShared;

#[cube]
impl<EIn: JitElement, EOut: JitElement> ReduceDimShared<EIn, EOut> for Argmax {
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

        if value > values[write_position] {
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
