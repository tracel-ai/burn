use super::Kernel;
use crate::{tensor::JitTensor, JitElement, JitRuntime};
use burn_cube::{calculate_cube_count_elemwise, prelude::*};
use burn_cube::{frontend::TensorArg, KernelSettings, SUBCUBE_DIM_APPROX};

/// Returns the offset of the tensor corresponding to the layout tensor.
#[cube]
pub fn index_offset_with_layout<N: CubePrimitive>(
    tensor: &Tensor<N>,
    layout: &Tensor<N>,
    offset_layout: UInt,
    dim_start: UInt,
    dim_end: UInt,
    unroll: Comptime<bool>,
) -> UInt {
    let vectorization_factor = Comptime::vectorization(tensor);
    let vectorization_factor_runtime = Comptime::runtime(vectorization_factor);

    let offset_ref = offset_layout * vectorization_factor_runtime;
    let mut offset = UInt::new(0);

    for i in range(dim_start, dim_end, unroll) {
        let ogwl = offset_ref / layout.stride(i);
        offset += ogwl % tensor.shape(i) * tensor.stride(i);
    }

    offset / vectorization_factor_runtime
}

#[cube(launch)]
fn into_contiguous_kernel<N: CubePrimitive>(
    input: &Tensor<N>,
    output: &mut Tensor<N>,
    rank: Comptime<Option<UInt>>,
) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        return;
    }

    let offset_input = index_offset_with_layout::<N>(
        input,
        output,
        offset_output,
        UInt::new(0),
        Comptime::unwrap_or_else(rank, || output.rank()),
        Comptime::is_some(rank),
    );

    output[offset_output] = input[offset_input];
}

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: JitRuntime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    if tensor.is_contiguous() {
        return tensor;
    }

    // Vectorization is only enabled when the last dimension is contiguous.
    let vectorization_factor = if tensor.strides[D - 1] == 1 {
        let last_dim = tensor.shape.dims[D - 1];
        if last_dim % 4 == 0 {
            4
        } else if last_dim % 2 == 0 {
            2
        } else {
            1
        }
    } else {
        1
    };

    let client = tensor.client.clone();
    let num_elems = tensor.shape.num_elements();
    let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
    let output = JitTensor::new(
        tensor.client.clone(),
        tensor.device,
        tensor.shape.clone(),
        buffer,
    );
    let cube_count = calculate_cube_count_elemwise(
        num_elems / vectorization_factor as usize,
        SUBCUBE_DIM_APPROX,
    );

    into_contiguous_kernel_launch::<E::Primitive, R>(
        client,
        cube_count,
        CubeDim::default(),
        TensorArg::vectorized(
            vectorization_factor,
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        ),
        TensorArg::vectorized(
            vectorization_factor,
            &output.handle,
            &output.strides,
            &output.shape.dims,
        ),
        Some(UInt::new(D as u32)),
    );

    output
}
