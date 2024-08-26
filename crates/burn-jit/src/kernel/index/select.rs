use crate::{
    element::JitElement, kernel::Kernel, ops::numeric::empty_device, tensor::JitTensor, JitRuntime,
};
use cubecl::prelude::*;
use cubecl::{calculate_cube_count_elemwise, CubeDim};

#[cube(launch_unchecked)]
fn select_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<T>,
    indices: &Tensor<I>,
    output: &mut Tensor<T>,
    dim: UInt,
) {
    if ABSOLUTE_POS >= output.len() {
        return;
    }

    let mut offset_input = UInt::new(0);

    for i in range(0u32, output.rank(), Comptime::new(false)) {
        let mut offset_local = ABSOLUTE_POS / output.stride(i) % output.shape(i);

        if i == dim {
            offset_local = UInt::cast_from(indices[offset_local]);
        }

        offset_input += offset_local * input.stride(i);
    }

    output[ABSOLUTE_POS] = input[offset_input];
}

pub(crate) fn select<R: JitRuntime, E: JitElement, I: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
    indices: JitTensor<R, I, 1>,
) -> JitTensor<R, E, D> {
    let mut shape_output = tensor.shape.clone();
    shape_output.dims[dim] = indices.shape.dims[0];
    let total_elem = shape_output.num_elements();

    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);

    let dummy_array = [1; D];
    let cube_dim = CubeDim::default();
    let cube_count = calculate_cube_count_elemwise(total_elem, cube_dim);

    unsafe {
        select_kernel::launch_unchecked::<E::Primitive, I::Primitive, R>(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            // Ignore shape and stride
            TensorArg::from_raw_parts(&indices.handle, &dummy_array, &dummy_array, 1),
            output.as_tensor_arg(1),
            ScalarArg::new(dim as u32),
        )
    };
    output
}
