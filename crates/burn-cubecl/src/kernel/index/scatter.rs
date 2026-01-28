use crate::{
    CubeRuntime,
    kernel::{AddOp, BinaryOp, BinaryOpFamily, OrOp, utils::shape_divmod},
    tensor::CubeTensor,
};
use cubecl::{CubeDim, calculate_cube_count_elemwise};
use cubecl::{prelude::*, std::FastDivmod};

#[cube(launch_unchecked)]
fn scatter_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(
    input: &mut Tensor<T>,
    indices: &Tensor<I>,
    value: &Tensor<T>,
    in_shape: Sequence<FastDivmod<usize>>,
    #[comptime] dim: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    let rank = in_shape.len().comptime();
    let stride_input = input.stride(dim);
    let stride_value = value.stride(dim);
    let stride_indices = indices.stride(dim);
    let shape_value = value.shape(dim);

    let mut offset = ABSOLUTE_POS;
    let mut offset_input = 0;
    let mut offset_indices = 0;
    let mut offset_value = 0;
    let mut num_elems = 1;

    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        if i != dim {
            let shape_input_loop = input.shape(i);

            let (rem, local_pos) = in_shape[i].div_mod(offset);
            offset = rem;

            offset_input += local_pos * input.stride(i);
            offset_indices += local_pos * indices.stride(i);
            offset_value += local_pos * value.stride(i);

            num_elems *= shape_input_loop;
        }
    }

    let should_stop = ABSOLUTE_POS >= num_elems;
    if should_stop {
        terminate!();
    }

    for i in 0..shape_value {
        let value_idx = (stride_value * i) + offset_value;
        let index_idx = (stride_indices * i) + offset_indices;

        let value = value[value_idx];
        let index = usize::cast_from(indices[index_idx]);

        let input_idx = (stride_input * index) + offset_input;

        let value =
            Op::BinaryOp::<T>::execute(Line::cast_from(input[input_idx]), Line::cast_from(value));
        input[input_idx] = value[0];
    }
}

pub(crate) fn scatter<R: CubeRuntime>(
    dim: usize,
    tensor: CubeTensor<R>,
    indices: CubeTensor<R>,
    value: CubeTensor<R>,
    is_bool: bool,
) -> CubeTensor<R> {
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let num_elems = tensor.shape.num_elements() / tensor.shape.dims[dim];

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
            shape_divmod(&tensor),
            dim,
            [tensor.dtype.into(), indices.dtype.into()],
        )
        .expect("Kernel to never fail");
    }
    tensor
}
