use crate::{
    CubeRuntime,
    kernel::{
        AddOp, AssignOp, BinaryMaxOp, BinaryMinOp, BinaryOp, BinaryOpFamily, MulOp,
        utils::address_type,
    },
    tensor::CubeTensor,
};
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::TensorMetadata;
use cubecl::prelude::*;
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

/// scatter_nd GPU kernel.
///
/// Each thread handles one element across all update slices.
/// Work items = num_updates * slice_size.
#[cube(launch_unchecked, address_type = "dynamic")]
fn scatter_nd_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(
    data: &mut Tensor<T>,
    indices: &LinearView<I>,
    values: &LinearView<T>,
    slice_size: usize,
    k: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if !values.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    // Decompose ABSOLUTE_POS into (update_index, slice_offset)
    let slice_offset = ABSOLUTE_POS % slice_size;
    let update_idx = ABSOLUTE_POS / slice_size;

    // Compute flat base offset into data from the K-dimensional index tuple
    let idx_base = update_idx * k;
    let mut base_offset = 0usize;
    for j in 0..k {
        let idx_val = usize::cast_from(indices[idx_base + j]);
        base_offset += idx_val * data.stride(j);
    }

    let data_idx = base_offset + slice_offset;
    let val_idx = ABSOLUTE_POS;

    let result = Op::BinaryOp::<T, Const<1>>::execute(
        Vector::cast_from(data[data_idx]),
        Vector::cast_from(values[val_idx]),
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

    let data_shape = tensor.shape();
    let idx_shape = indices.shape();
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

    unsafe {
        launch(
            &tensor.client.clone(),
            cube_count,
            cube_dim,
            address_type!(tensor, indices, values),
            tensor.clone().into_tensor_arg(),
            indices.into_linear_view(),
            values.into_linear_view(),
            slice_size,
            k,
            [tensor_dtype.into(), indices_dtype.into()],
        )
    }

    tensor
}
