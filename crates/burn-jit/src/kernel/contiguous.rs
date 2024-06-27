use super::Kernel;
use crate::{tensor::JitTensor, JitElement, JitRuntime};
use burn_cube::{calculate_cube_count_elemwise, prelude::*};
use burn_cube::{frontend::TensorHandle, KernelSettings, SUBCUBE_DIM_APPROX};

#[cube]
fn index_offset_global_with_layout<N: CubeElem>(
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
fn into_contiguous_kernel<N: CubeElem>(input: &Tensor<N>, output: &mut Tensor<N>) {
    let offset_output = ABSOLUTE_POS;

    if offset_output >= output.len() {
        return;
    }

    let offset_input = index_offset_global_with_layout::<N>(
        &input,
        &output,
        offset_output,
        UInt::new(0),
        input.rank(),
        Comptime::new(false),
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
    let settings = KernelSettings::default()
        .vectorize_input(0, vectorization_factor)
        .vectorize_output(0, vectorization_factor);
    let cube_count = calculate_cube_count_elemwise(
        num_elems / vectorization_factor as usize,
        SUBCUBE_DIM_APPROX,
    );

    into_contiguous_kernel_launch::<E::CubeElement, R>(
        client,
        cube_count,
        settings,
        TensorHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
        TensorHandle::new(&output.handle, &output.strides, &output.shape.dims),
    );

    output
}
