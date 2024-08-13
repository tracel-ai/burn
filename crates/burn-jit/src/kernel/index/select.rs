use crate::{
    element::JitElement, kernel::Kernel, ops::numeric::empty_device, tensor::JitTensor, JitRuntime,
};
use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, CubeDim};

#[cube(launch_unchecked)]
fn select_kernel<T: Numeric>(
    input: &Tensor<T>,
    indices: &Tensor<I32>,
    output: &mut Tensor<T>,
    dim: &UInt,
) {
    let id = ABSOLUTE_POS;
    let mut offset_input = UInt::new(0);
    let rank = output.rank();
    for i in range(UInt::new(0), rank, Comptime::new(false)) {
        let stride_input = input.stride(i);
        let stride_output = output.stride(i);
        let shape_output = output.shape(i);
        let mut offset_local = id / stride_output;
        offset_local = offset_local % shape_output;

        if i == *dim {
            offset_local = UInt::cast_from(indices[offset_local]);
            offset_local *= stride_input;
        } else {
            offset_local *= stride_input;
        }
        offset_input += offset_local;
    }
    let value = input[offset_input];
    output[id] = value;
}

pub(crate) fn select<R: JitRuntime, E: JitElement, I: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
    indices: JitTensor<R, I, 1>,
) -> JitTensor<R, E, D> {
    let mut shape_output = tensor.shape.clone();
    let num_elems = indices.shape.dims[0];
    shape_output.dims[dim] = num_elems;

    let mut total_elem = 1;
    for dim_size in shape_output.dims.iter() {
        total_elem *= dim_size
    }

    let mut shapes = [1; D];
    let mut strides = [num_elems; D];
    shapes[D - 1] = num_elems;
    strides[D - 1] = 1;

    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);

    let cube_dim = CubeDim::default();

    let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);

    unsafe {
        select_kernel::launch_unchecked::<E::Primitive, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            TensorArg::from_raw_parts(&indices.handle, &strides, &shapes, 1),
            output.as_tensor_arg(1),
            ScalarArg::new(dim as u32),
        )
    };

    output
}
