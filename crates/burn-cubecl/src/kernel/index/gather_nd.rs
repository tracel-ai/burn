use crate::{
    CubeRuntime, kernel::utils::address_type, ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_backend::TensorMetadata;
use cubecl::prelude::*;
use cubecl::std::tensor::layout::linear::LinearView;
use cubecl::{CubeDim, calculate_cube_count_elemwise};

/// gather_nd GPU kernel.
///
/// Each thread handles one element of the output.
/// Work items = num_indices * slice_size.
#[cube(launch_unchecked, address_type = "dynamic")]
fn gather_nd_kernel<T: Numeric, I: Int>(
    data: &Tensor<T>,
    indices: &LinearView<I>,
    output: &mut LinearView<T, ReadWrite>,
    slice_size: usize,
    k: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if !output.is_in_bounds(ABSOLUTE_POS) {
        terminate!();
    }

    // Decompose ABSOLUTE_POS into (index_tuple_idx, slice_offset)
    let slice_offset = ABSOLUTE_POS % slice_size;
    let index_idx = ABSOLUTE_POS / slice_size;

    // Compute flat offset into data from the K-dimensional index tuple
    let idx_base = index_idx * k;
    let mut base_offset = 0usize;
    for j in 0..k {
        let idx_val = usize::cast_from(indices[idx_base + j]);
        base_offset += idx_val * data.stride(j);
    }

    output[ABSOLUTE_POS] = data[base_offset + slice_offset];
}

pub(crate) fn gather_nd<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    indices: CubeTensor<R>,
) -> CubeTensor<R> {
    let data_shape = tensor.shape();
    let idx_shape = indices.shape();
    let m = idx_shape.num_dims();
    let k = idx_shape[m - 1];

    // num_indices = product of first M-1 dims of indices
    let num_indices: usize = idx_shape.as_slice()[..m - 1].iter().product();
    // slice_size = product of data.shape[K..]
    let slice_size: usize = data_shape.as_slice()[k..].iter().product();
    let total_elem = num_indices * slice_size;

    // Output shape: idx_shape[..m-1] ++ data_shape[k..]
    let mut out_dims: Vec<usize> = idx_shape.as_slice()[..m - 1].to_vec();
    out_dims.extend_from_slice(&data_shape.as_slice()[k..]);
    let out_shape = burn_backend::Shape::from(out_dims);

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        out_shape,
        tensor.dtype,
    );

    let cube_dim = CubeDim::new(&tensor.client, total_elem);
    let cube_count = calculate_cube_count_elemwise(&tensor.client, total_elem, cube_dim);

    let (dtype, indices_dtype) = (tensor.dtype, indices.dtype);

    unsafe {
        gather_nd_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(tensor, indices, output),
            tensor.into_tensor_arg(),
            indices.into_linear_view(),
            output.clone().into_linear_view(),
            slice_size,
            k,
            [dtype.into(), indices_dtype.into()],
        )
    }

    output
}
