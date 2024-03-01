use crate::{
    codegen::dialect::gpu::{Item, Scope, Variable},
    element::JitElement,
    tensor::JitTensor,
    Runtime,
};

pub(crate) trait ReduceDimAlgorithm: Send + Sync + 'static {
    type Accumulator: Copy;

    fn initialize_naive(
        scope: &mut Scope,
        input_item: Item,
        output_item: Item,
    ) -> Self::Accumulator;

    fn inner_loop_naive(
        scope: &mut Scope,
        accumulator: Self::Accumulator,
        current_value: Variable,
        i: Variable,
    );

    fn assign_naive(
        scope: &mut Scope,
        output: Variable,
        accumulator: Self::Accumulator,
        shape_reduce_dim: Variable,
    );

    fn initialize_shared(
        scope: &mut Scope,
        shared_memory_size: u32,
        write_position: Variable,
        input_item: Item,
    ) -> Self::Accumulator;

    fn write_to_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        write_position: Variable,
        value: Self::Accumulator,
    );

    fn read_from_input(
        scope: &mut Scope,
        input: Variable,
        read_position: Variable,
        i: Variable,
    ) -> Self::Accumulator;

    fn read_from_shared(
        scope: &mut Scope,
        shared_memory: Self::Accumulator,
        read_position: Variable,
    ) -> Self::Accumulator;

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
