use crate::{kernel::reduce::SumDim, JitElement};
use cubecl::prelude::*;

use super::base::ReduceDimShared;

#[cube]
impl<EIn: JitElement, EOut: JitElement> ReduceDimShared<EIn, EOut> for SumDim {
    /// The reduction accumulator
    type Accumulator = SharedMemory<EIn>;
    type Value = EIn;

    /// Initialization for shared algorithm
    fn initialize_shared(shared_memory_size: u32, write_position: u32) -> SharedMemory<EIn> {
        let mut value_shared = SharedMemory::new(shared_memory_size);
        value_shared[write_position] = EIn::from_int(0);
        value_shared
    }

    /// How to write to shared memory
    fn write_to_shared(shared_memory: &mut SharedMemory<EIn>, write_position: u32, value: EIn) {
        shared_memory[write_position] += value;
    }

    /// How to read from input in shared algorithm
    fn read_from_input(input: &Tensor<EIn>, read_position: u32, _i: u32) -> EIn {
        input[read_position]
    }

    /// How to read from shared memory
    fn read_from_shared(shared_memory: &SharedMemory<EIn>, read_position: u32) -> EIn {
        shared_memory[read_position]
    }

    /// How to assign from shared memory
    fn assign_shared(
        shared_memory: &SharedMemory<EIn>,
        output: &mut Tensor<EOut>,
        write_position: u32,
        _shape_reduce_dim: u32,
    ) {
        output[write_position] = EOut::cast_from(shared_memory[0]);
    }
}
