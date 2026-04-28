use crate::kernel::utils::{shape_divmod, shape_divmod_range};
use crate::{
    CubeRuntime, kernel::utils::address_type, ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use cubecl::prelude::*;
use cubecl::std::FastDivmod;
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
    output: &mut Tensor<T>,
    output_shape: Sequence<FastDivmod<usize>>,
    data_slice_shape: Sequence<FastDivmod<usize>>,
    slice_size: usize,
    k: usize,
    working_units: usize,
    #[define(T, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let slice_offset = ABSOLUTE_POS % slice_size;
    let index_idx = ABSOLUTE_POS / slice_size;

    // Compute flat offset into data from the K-dimensional index tuple
    let idx_base = index_idx * k;
    let mut base_offset = 0usize;
    for j in 0..k {
        let idx_val = usize::cast_from(indices[idx_base + j]);
        base_offset += idx_val * data.stride(j);
    }

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

    let out_rank = output_shape.len().comptime();
    let mut out_offset = 0usize;
    let mut remainder_o = ABSOLUTE_POS;
    #[unroll]
    for i in 0..out_rank {
        let dim = out_rank - i - 1;
        let (rem, coord) = output_shape[dim].div_mod(remainder_o);
        remainder_o = rem;
        out_offset += coord * output.stride(dim);
    }

    output[out_offset] = data[base_offset + data_slice_offset];
}

pub(crate) fn gather_nd<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    indices: CubeTensor<R>,
) -> CubeTensor<R> {
    let data_shape = &tensor.meta.shape;
    let idx_shape = &indices.meta.shape;
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

    let data_slice_shape = shape_divmod_range(&tensor, k..data_shape.num_dims());
    let output_shape_divmod = shape_divmod(&output);

    unsafe {
        gather_nd_kernel::launch_unchecked(
            &output.client,
            cube_count,
            cube_dim,
            address_type!(tensor, indices, output),
            tensor.into_tensor_arg(),
            indices.into_linear_view(),
            output.clone().into_tensor_arg(),
            output_shape_divmod,
            data_slice_shape,
            slice_size,
            k,
            total_elem,
            [dtype.into(), indices_dtype.into()],
        )
    }

    output
}
