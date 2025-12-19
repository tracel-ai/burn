use crate::kernel::into_contiguous;
use crate::ops::numeric::empty_device_dtype;
use crate::{CubeRuntime, tensor::CubeTensor};
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

#[cube(launch_unchecked)]
fn select_kernel<T: Numeric, I: Numeric>(
    input: &Tensor<T>,
    indices: &Tensor<I>,
    output: &mut Tensor<T>,
    dim: u32,
    #[define(T, I)] _dtypes: [StorageType; 2],
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

pub(crate) fn select<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    dim: usize,
    indices: CubeTensor<R>,
) -> CubeTensor<R> {
    let ndims = tensor.shape.num_dims();
    let mut shape_output = tensor.shape.clone();
    shape_output.dims[dim] = indices.shape[0];
    let total_elem = shape_output.num_elements();
    let indices = into_contiguous(indices);

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        shape_output,
        tensor.dtype,
    );

    let dummy_array = vec![1; ndims];
    let working_units = total_elem;
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    unsafe {
        select_kernel::launch_unchecked(
            &tensor.client,
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            // Ignore shape and stride
            TensorArg::from_raw_parts_and_size(
                &indices.handle,
                &dummy_array,
                &dummy_array,
                1,
                indices.dtype.size(),
            ),
            output.as_tensor_arg(1),
            ScalarArg::new(dim as u32),
            [tensor.dtype.into(), indices.dtype.into()],
        )
        .expect("Kernel to never fail");
    };
    output
}
