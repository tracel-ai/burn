use crate::{
    CubeRuntime,
    kernel::{
        AddOp, AssignOp, BinaryMaxOp, BinaryMinOp, BinaryOp, BinaryOpFamily, MulOp,
        utils::address_type,
    },
    tensor::CubeTensor,
};
use burn_backend::tensor::IndexingUpdateOp;
use burn_backend::{DType, TensorMetadata};
use burn_std::Metadata;
use cubecl::prelude::*;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

/// Compute the strides used to convert K-dimensional index tuples into flat
/// offsets into a contiguous data tensor, returned as a 1D CubeTensor of u32.
///
/// Entry `j` = product of `data_shape[j+1..]` truncated to the first `k`
/// dimensions, with the innermost stride equal to `slice_size`.
pub(super) fn nd_index_strides_tensor<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    data_shape: &burn_backend::Shape,
    k: usize,
    slice_size: usize,
) -> CubeTensor<R> {
    let mut strides = vec![0u32; k];
    if k > 0 {
        strides[k - 1] = slice_size as u32;
        for i in (0..k - 1).rev() {
            strides[i] = strides[i + 1] * data_shape[i + 1] as u32;
        }
    }
    let data = burn_backend::TensorData::new(strides, burn_backend::Shape::from([k]));
    let shape = burn_backend::Shape::from([k]);
    let alloc = tensor
        .client
        .create_tensor(data.bytes, shape.clone(), data.dtype.size());
    CubeTensor::new(
        tensor.client.clone(),
        alloc.memory,
        Metadata::new(shape, alloc.strides),
        tensor.device.clone(),
        DType::U32,
    )
}

/// scatter_nd GPU kernel.
///
/// Each thread handles one element across all update slices.
/// Work items = num_updates * slice_size.
///
/// `data_strides` is a 1D tensor of length K holding the strides for converting
/// K-dimensional index tuples into flat offsets.
#[cube(launch_unchecked, address_type = "dynamic")]
fn scatter_nd_kernel<T: Numeric, I: Int, Op: BinaryOpFamily>(
    data: &mut Tensor<T>,
    indices: &Tensor<I>,
    values: &Tensor<T>,
    data_strides: &Tensor<u32>,
    slice_size: usize,
    k: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    let num_updates_times_slice = values.len();
    if ABSOLUTE_POS >= num_updates_times_slice {
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
        base_offset += idx_val * data_strides[j] as usize;
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

    let strides_tensor = nd_index_strides_tensor(&tensor, &data_shape, k, slice_size);

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
            address_type!(tensor, indices, values, strides_tensor),
            tensor.clone().into_tensor_arg(),
            indices.into_tensor_arg(),
            values.into_tensor_arg(),
            strides_tensor.into_tensor_arg(),
            slice_size,
            k,
            [tensor_dtype.into(), indices_dtype.into()],
        )
    }

    tensor
}
