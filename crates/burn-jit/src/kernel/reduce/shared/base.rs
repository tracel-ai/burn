use cubecl::prelude::*;

use crate::JitElement;

/// Specifies the reduce dim algorithm in use
#[cube]
pub trait ReduceDimShared<EIn: JitElement, EOut: JitElement>: Send + Sync + 'static {
    /// The reduction accumulator
    type Accumulator: CubeType;
    type Value: CubeType;

    /// Initialization for shared algorithm
    fn initialize_shared(shared_memory_size: u32, write_position: u32) -> Self::Accumulator;

    /// How to write to shared memory
    fn write_to_shared(
        shared_memory: &mut Self::Accumulator,
        write_position: u32,
        value: Self::Value,
    );

    /// How to read from input in shared algorithm
    fn read_from_input(input: &Tensor<EIn>, read_position: u32, i: u32) -> Self::Value;

    /// How to read from shared memory
    fn read_from_shared(shared_memory: &Self::Accumulator, read_position: u32) -> Self::Value;

    /// How to assign from shared memory
    fn assign_shared(
        shared_memory: &Self::Accumulator,
        output: &mut Tensor<EOut>,
        write_position: u32,
        shape_reduce_dim: u32,
    );
}
