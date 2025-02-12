use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};
use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, CubeDim};

#[cube(launch_unchecked)]
fn select_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<T>,
    indices: &Tensor<I>,
    output: &mut Tensor<T>,
    dim: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mut offset_input = 0;

    for i in 0..output.rank() {
        let mut offset_local = ABSOLUTE_POS / output.stride(i) % output.shape(i);

        if i == dim {
            offset_local = u32::cast_from(indices[offset_local]);
        }

        offset_input += offset_local * input.stride(i);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub(crate) fn select<R: JitRuntime, E: JitElement, I: JitElement>(
    tensor: JitTensor<R>,
    dim: usize,
    indices: JitTensor<R>,
) -> JitTensor<R> {
    let ndims = tensor.shape.num_dims();
    let mut shape_output = tensor.shape.clone();
    shape_output.dims[dim] = indices.shape.dims[0];
    let total_elem = shape_output.num_elements();

    let output = empty_device::<R, E>(tensor.client.clone(), tensor.device.clone(), shape_output);

    let dummy_array = vec![1; ndims];
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);

    unsafe {
        select_kernel::launch_unchecked::<E, I, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg::<E>(1),
            // Ignore shape and stride
            TensorArg::from_raw_parts::<I>(&indices.handle, &dummy_array, &dummy_array, 1),
            output.as_tensor_arg::<E>(1),
            ScalarArg::new(dim as u32),
        )
    };
    output
}
