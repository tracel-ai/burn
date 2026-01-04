use crate::{
    CubeRuntime,
    kernel::{self, AddOp, BinaryOp, BinaryOpFamily, OrOp},
    tensor::CubeTensor,
};
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

#[cube(launch_unchecked)]
fn scatter_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(
    input: &mut Tensor<T>,
    indices: &Tensor<I>,
    value: &Tensor<T>,
    dim: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    let stride_input = input.stride(dim);
    let shape_value = value.shape(dim);

    let mut offset_input = 0;
    let mut offset_value = 0;
    let mut num_elems = 1;

    for i in 0..value.rank() {
        let shouldnt_skip = i != dim;
        if shouldnt_skip {
            let shape_input_loop = input.shape(i);
            let shape_value_loop = value.shape(i);

            let stride_value_loop = value.stride(i);
            let stride_input_loop = input.stride(i);
            let stride_tmp = indices.stride(i);

            let mut num_blocks = ABSOLUTE_POS / stride_tmp;
            num_blocks %= shape_input_loop;

            let mut offset_tmp = num_blocks * stride_input_loop;
            offset_input += offset_tmp;

            offset_tmp = num_blocks * stride_value_loop;
            offset_value += offset_tmp;

            num_elems *= shape_value_loop;
        }
    }

    let should_stop = ABSOLUTE_POS >= num_elems;
    if should_stop {
        terminate!();
    }

    for i in 0..shape_value {
        let mut idx = stride_input * i;
        idx += offset_value;

        let result_value = value[idx];
        let result_indices = usize::cast_from(indices[idx]);

        let mut index_input = stride_input * result_indices;
        index_input += offset_input;

        let value = Op::BinaryOp::<T>::execute(
            Line::cast_from(input[index_input]),
            Line::cast_from(result_value),
        );
        input[index_input] = value[0];
    }
}

pub(crate) fn scatter<R: CubeRuntime>(
    dim: usize,
    tensor: CubeTensor<R>,
    indices: CubeTensor<R>,
    value: CubeTensor<R>,
    is_bool: bool,
) -> CubeTensor<R> {
    let ndims = tensor.shape.num_dims();
    let mut indices = kernel::into_contiguous(indices);
    let tensor = kernel::into_contiguous(tensor);
    let value = kernel::into_contiguous(value);

    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let mut strides = vec![0; ndims];
    let mut current = 1;
    let mut num_elems = 1;

    tensor
        .shape
        .dims
        .iter()
        .enumerate()
        .rev()
        .filter(|(index, _val)| *index != dim)
        .for_each(|(index, val)| {
            strides[index] = current;
            current *= val;
            num_elems *= tensor.shape[index];
        });

    // Fake strides of the virtual output where the strides of dim is hardcoded to one.
    indices.strides = strides;

    let working_units = num_elems;
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    let launch = match is_bool {
        true => scatter_kernel::launch_unchecked::<OrOp, R>,
        false => scatter_kernel::launch_unchecked::<AddOp, R>,
    };

    unsafe {
        launch(
            &indices.client.clone(),
            cube_count,
            cube_dim,
            tensor.as_tensor_arg(1),
            indices.as_tensor_arg(1),
            value.as_tensor_arg(1),
            ScalarArg::new(dim),
            [tensor.dtype.into(), indices.dtype.into()],
        )
        .expect("Kernel to never fail");
    }
    tensor
}
