use crate::{kernel::Kernel, tensor::JitTensor, JitElement, JitRuntime};
use cubecl::linalg::tensor::index_offset_with_layout;
use cubecl::{
    calculate_cube_count_elemwise, prelude::*, tensor_vectorization_factor, SUBCUBE_DIM_APPROX,
};
use cubecl::{
    cpa,
    frontend::TensorHandleRef,
    ir::{KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::{any::TypeId, marker::PhantomData};

#[cube(launch)]
pub(crate) fn cast_element<I: CubePrimitive, O: CubePrimitive>(
    input: &Tensor<I>,
    output: &mut Tensor<O>,
    rank: Comptime<Option<UInt>>,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        return;
    }

    let offset_input = index_offset_with_layout::<I, O>(
        input,
        output,
        offset_output,
        UInt::new(0),
        Comptime::unwrap_or_else(rank, || output.rank()),
        Comptime::is_some(rank),
    );

    output[offset_output] = O::cast_from(input[offset_input]);
}

/// Cast a tensor to the given element type.
///
/// Note: When input element is semantically a boolean, prefer bool_cast function.
pub fn cast<R: JitRuntime, EI: JitElement, EO: JitElement, const D: usize>(
    input: JitTensor<R, EI, D>,
) -> JitTensor<R, EO, D> {
    if TypeId::of::<EI>() == TypeId::of::<EO>() {
        return JitTensor::new_contiguous(input.client, input.device, input.shape, input.handle);
    }

    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = D;
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &input.shape.dims, &input.strides, rank - 1);

    let num_elems: usize = input.shape.num_elements();
    let cube_count = calculate_cube_count_elemwise(
        num_elems / vectorization_factor as usize,
        SUBCUBE_DIM_APPROX,
    );
    let client = input.client.clone();
    let handle = client.empty(num_elems * core::mem::size_of::<EO>());
    let output = JitTensor::new_contiguous(client.clone(), input.device, input.shape.clone(), handle);

    cast_element::launch::<EI::Primitive, EO::Primitive, R>(
        &client,
        cube_count,
        CubeDim::default(),
        TensorArg::vectorized(
            vectorization_factor,
            &input.handle,
            &input.strides,
            &input.shape.dims,
        ),
        TensorArg::vectorized(
            vectorization_factor,
            &output.handle,
            &output.strides,
            &output.shape.dims,
        ),
        Some(UInt::new(rank as u32)),
    );

    output
}
