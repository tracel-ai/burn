use crate::kernel::{
    AddOp, BinaryOp, BinaryOpFamily, OrOp,
    utils::{address_type, linear_view, shape_divmod},
};
use crate::{CubeRuntime, tensor::CubeTensor};
use cubecl::{CubeDim, calculate_cube_count_elemwise, std::tensor::layout::linear::LinearView};
use cubecl::{prelude::*, std::FastDivmod};

#[cube(launch_unchecked, address_type = "dynamic")]
fn select_assign_kernel<F: Numeric, I: Numeric, Op: BinaryOpFamily>(
    tensor: &mut Tensor<F>,
    indices: &LinearView<I>,
    value: &Tensor<F>,
    value_shape: Sequence<FastDivmod<usize>>,
    num_elems: usize,
    #[comptime] dim: usize,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= num_elems {
        terminate!();
    }

    let rank = value_shape.len().comptime();

    let mut offset = ABSOLUTE_POS;
    let mut offset_tensor = 0;
    let mut offset_value = 0;

    // Calculate offsets and num_elems
    #[unroll]
    for i in 0..rank {
        let i = rank - i - 1;
        if i != dim {
            let (rem, local_pos) = value_shape[i].div_mod(offset);
            offset = rem;

            offset_tensor += local_pos * tensor.stride(i);
            offset_value += local_pos * value.stride(i);
        }
    }

    let strides_tensor_dim = tensor.stride(dim);
    let strides_value_dim = value.stride(dim);

    // Main operation
    for i in 0..value.shape(dim) {
        let index_tensor = usize::cast_from(indices[i]) * strides_tensor_dim + offset_tensor;
        let index_value = i * strides_value_dim + offset_value;

        let value = Op::BinaryOp::<F>::execute(
            Line::cast_from(tensor[index_tensor]),
            Line::cast_from(value[index_value]),
        );
        tensor[index_tensor] = F::cast_from(value);
    }
}

pub(crate) fn select_assign<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    dim: usize,
    indices: CubeTensor<R>,
    value: CubeTensor<R>,
    is_bool: bool,
) -> CubeTensor<R> {
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let num_elems = tensor.meta.num_elements() / tensor.meta.shape()[dim];
    let working_units = num_elems;
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    let launch = match is_bool {
        true => select_assign_kernel::launch_unchecked::<OrOp, R>,
        false => select_assign_kernel::launch_unchecked::<AddOp, R>,
    };

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    let shape = shape_divmod(&value);
    unsafe {
        launch(
            &tensor.client,
            cube_count,
            cube_dim,
            address_type!(tensor, indices, value),
            tensor.clone().into_tensor_arg(1),
            linear_view(indices, 1),
            value.into_tensor_arg(1),
            shape,
            ScalarArg::new(num_elems),
            dim,
            [tensor_dtype.into(), indices_dtype.into()],
        )
    };

    tensor
}
