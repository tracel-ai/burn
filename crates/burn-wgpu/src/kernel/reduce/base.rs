use crate::{
    codegen::dialect::gpu::{Item, Scope, Variable},
    element::JitElement,
    tensor::JitTensor,
    Runtime,
};

/// Specifies the reduce dim algorithm in use
pub trait ReduceDimAlgorithm: Send + Sync + 'static {
    /// The reduction accumulator
    type Accumulator: Copy;

    /// Initialization for naive algorithm
    fn initialize_naive(
        scope: &mut Scope,
        input_item: Item,
        output_item: Item,
    ) -> Self::Accumulator;

    /// Inner loop for naive algorithm
    fn inner_loop_naive(
        scope: &mut Scope,
        accumulator: Self::Accumulator,
        current_value: Variable,
        i: Variable,
    );

    /// Assignation for naive algorithm
    fn assign_naive(
        scope: &mut Scope,
        output: Variable,
        accumulator: Self::Accumulator,
        shape_reduce_dim: Variable,
    );

    /// Initialization for shared algorithm
    fn initialize_shared(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator;

    /// How to write to shared memory
    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        value: Self::Accumulator,
    );

    /// How to read from input in shared algorithm
    fn read_from_input(
        scope: &mut Scope,
        input: Variable,
        read_position: Variable,
        i: Variable,
    ) -> Self::Accumulator;

    /// How to read from shared memory
    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Self::Accumulator;

    /// How to assign from shared memory
    fn assign_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        output: Variable,
        write_position: Variable,
        shape_reduce_dim: Variable,
    );
}

/// Creates an empty output tensor with reduce output shape
pub fn init_reduce_output<R: Runtime, EI: JitElement, EO: JitElement, const D: usize>(
    input: &JitTensor<R, EI, D>,
    reduce_dim: usize,
) -> JitTensor<R, EO, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[reduce_dim] = 1;

    // Create output handle
    let num_elems_output = shape_out.num_elements();
    let handle = input
        .client
        .empty(num_elems_output * core::mem::size_of::<EO>());
    JitTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
        handle,
    )
}
