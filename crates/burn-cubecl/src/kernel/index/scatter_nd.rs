use crate::{
    CubeRuntime,
    kernel::{
        AddOp, AssignOp, BinaryMaxOp, BinaryMinOp, BinaryOp, BinaryOpFamily, MulOp,
        utils::{address_type, shape_divmod_range},
    },
    tensor::CubeTensor,
};
use burn_backend::tensor::IndexingUpdateOp;
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{CubeDim, calculate_cube_count_elemwise};
use cubecl::{prelude::*, std::FastDivmod};

/// scatter_nd GPU kernel.
///
/// Each thread handles one element across all update slices.
/// Work items = num_updates * slice_size.
#[cube(launch_unchecked, address_type = "dynamic")]
fn scatter_nd_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(
    data: &mut Tensor<T>,
    indices: &LinearView<I>,
    values: &Tensor<T>,
    data_slice_shape: Sequence<FastDivmod<usize>>,
    values_shape: Sequence<FastDivmod<usize>>,
    slice_size: usize,
    k: usize,
    working_units: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let slice_offset = ABSOLUTE_POS % slice_size;
    let update_idx = ABSOLUTE_POS / slice_size;

    let idx_base = update_idx * k;
    let mut base_offset = 0usize;
    for j in 0..k {
        let idx_val = usize::cast_from(indices[idx_base + j]);
        base_offset += idx_val * data.stride(j);
    }

    // Decompose slice_offset over data's trailing dims (k..n)
    let slice_rank = data_slice_shape.len().comptime();
    let mut data_slice_offset = 0usize;
    let mut remainder = slice_offset;
    #[unroll]
    for i in 0..slice_rank {
        let dim = slice_rank - i - 1;
        let (rem, coord) = data_slice_shape[dim].div_mod(remainder);
        remainder = rem;
        data_slice_offset += coord * data.stride(k + dim);
    }

    // Decompose slice_offset over values' dims (1..n_v), with update_idx for dim 0
    let val_rank = values_shape.len().comptime();
    let mut val_offset = update_idx * values.stride(0);
    let mut remainder_v = slice_offset;
    #[unroll]
    for i in 0..val_rank {
        let dim = val_rank - i - 1;
        let (rem, coord) = values_shape[dim].div_mod(remainder_v);
        remainder_v = rem;
        val_offset += coord * values.stride(1 + dim);
    }

    let data_idx = base_offset + data_slice_offset;
    let result = Op::BinaryOp::<T, Const<1>>::execute(
        Vector::cast_from(data[data_idx]),
        Vector::cast_from(values[val_offset]),
    );
    data[data_idx] = result[0];
}

pub(crate) fn scatter_nd<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    indices: CubeTensor<R>,
    values: CubeTensor<R>,
    reduction: IndexingUpdateOp,
) -> CubeTensor<R> {
    // Ensure we can write in-place
    let tensor = match tensor.can_mut() && tensor.is_nonoverlapping() {
        true => tensor,
        false => tensor.copy(),
    };

    let data_shape = &tensor.meta.shape;
    let idx_shape = &indices.meta.shape;
    let m = idx_shape.num_dims();
    let k = idx_shape[m - 1];

    // num_updates = product of first M-1 dims of indices
    let num_updates: usize = idx_shape.as_slice()[..m - 1].iter().product();
    // slice_size = product of data.shape[K..]
    let slice_size: usize = data_shape.as_slice()[k..].iter().product();
    let working_units = num_updates * slice_size;

    let cube_dim = CubeDim::new(&indices.client, working_units);
    let cube_count = calculate_cube_count_elemwise(&indices.client, working_units, cube_dim);

    let (tensor_dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    let launch = match reduction {
        IndexingUpdateOp::Assign => scatter_nd_kernel::launch_unchecked::<AssignOp, R>,
        IndexingUpdateOp::Add => scatter_nd_kernel::launch_unchecked::<AddOp, R>,
        IndexingUpdateOp::Mul => scatter_nd_kernel::launch_unchecked::<MulOp, R>,
        IndexingUpdateOp::Min => scatter_nd_kernel::launch_unchecked::<BinaryMinOp, R>,
        IndexingUpdateOp::Max => scatter_nd_kernel::launch_unchecked::<BinaryMaxOp, R>,
    };

    let data_slice_shape = shape_divmod_range(&tensor, k..data_shape.num_dims());
    // values dims 1.. (skip the num_updates leading dim)
    let values_slice_shape = shape_divmod_range(&values, 1..values.meta.shape.num_dims());

    unsafe {
        launch(
            &tensor.client.clone(),
            cube_count,
            cube_dim,
            address_type!(tensor, indices, values),
            tensor.clone().into_tensor_arg(),
            indices.into_linear_view(),
            values.into_tensor_arg(),
            data_slice_shape,
            values_slice_shape,
            slice_size,
            k,
            working_units,
            [tensor_dtype.into(), indices_dtype.into()],
        )
    }

    tensor
}
