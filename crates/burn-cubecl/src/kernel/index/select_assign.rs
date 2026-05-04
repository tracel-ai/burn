use crate::kernel::{
    AddOp, BinaryOp, BinaryOpFamily, OrOp,
    utils::{address_type, shape_divmod},
};
use crate::{CubeRuntime, tensor::CubeTensor};
use cubecl::{CubeDim, calculate_cube_count_elemwise, std::tensor::layout::linear::LinearView};
use cubecl::{prelude::*, std::FastDivmod};

/// Uses checked launch mode because user-provided `indices` may contain out-of-bounds values
/// that would cause invalid writes into `tensor`. Checked mode clamps these accesses rather
/// than producing undefined behavior.
#[cube(launch, address_type = "dynamic")]
fn select_assign_kernel<F: Numeric, I: Numeric, Op: BinaryOpFamily>(
    tensor: &mut Tensor<F>,
    indices: &LinearView<I>,
    value: &Tensor<F>,
    value_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[comptime] axis: usize,
    #[define(F, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= working_units {
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
        if i != axis {
            let (rem, local_pos) = value_shape[i].div_mod(offset);
            offset = rem;

            offset_tensor += local_pos * tensor.stride(i);
            offset_value += local_pos * value.stride(i);
        }
    }

    let strides_tensor_dim = tensor.stride(axis);
    let strides_value_dim = value.stride(axis);

    // Main operation
    for i in 0..value.shape(axis) {
        let index_tensor = usize::cast_from(indices[i]) * strides_tensor_dim + offset_tensor;
        let index_value = i * strides_value_dim + offset_value;

        let value = Op::BinaryOp::<F, Const<1>>::execute(
            Vector::cast_from(tensor[index_tensor]),
            Vector::cast_from(value[index_value]),
        );
        tensor[index_tensor] = F::cast_from(value);
    }
}

pub(crate) fn select_assign<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    axis: usize,
    indices: CubeTensor<R>,
    value: CubeTensor<R>,
    is_bool: bool,
) -> CubeTensor<R> {
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let working_units = value.meta.num_elements() / value.meta.shape()[axis];
    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    let launch = match is_bool {
        true => select_assign_kernel::launch::<OrOp, R>,
        false => select_assign_kernel::launch::<AddOp, R>,
    };

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    let shape = shape_divmod(&value);
    launch(
        &tensor.client,
        cube_count,
        cube_dim,
        address_type!(tensor, indices, value),
        tensor.clone().into_tensor_arg(),
        indices.into_linear_view(),
        value.into_tensor_arg(),
        shape,
        working_units,
        axis,
        [tensor_dtype.into(), indices_dtype.into()],
    );

    tensor
}
