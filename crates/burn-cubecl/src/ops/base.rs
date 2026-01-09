use crate::{CubeRuntime, kernel, tensor::CubeTensor};
use burn_backend::{
    DType, ExecutionError, QTensorPrimitive, Shape, TensorData,
    quantization::{QuantLevel, params_shape},
};
use burn_backend::{TensorMetadata, ops::unfold::calculate_unfold_shape};
use burn_std::tensor::{ReshapeAction, contiguous_strides, reshape_action};
use cubecl::{ir::LineSize, server::CopyDescriptor};
use cubecl::{quant::scheme::BlockSize, tensor_line_size_parallel};

pub(crate) fn from_data<R: CubeRuntime>(data: TensorData, device: &R::Device) -> CubeTensor<R> {
    let shape: Shape = (&data.shape).into();
    let client = R::client(device);
    let buffer = client.create(data.bytes);

    CubeTensor::new_contiguous(client, device.clone(), shape, buffer, data.dtype)
}

pub(crate) async fn into_data<R: CubeRuntime>(
    tensor: CubeTensor<R>,
) -> Result<TensorData, ExecutionError> {
    let tensor = kernel::into_contiguous_aligned(tensor);

    let elem_size = tensor.elem_size();
    let shape = &tensor.shape.dims;
    let binding = CopyDescriptor::new(tensor.handle.binding(), shape, &tensor.strides, elem_size);
    let bytes = tensor
        .client
        .read_one_tensor_async(binding)
        .await
        .map_err(|err| ExecutionError::WithContext {
            reason: format!("{err}"),
        })?;

    Ok(TensorData::from_bytes(bytes, tensor.shape, tensor.dtype))
}

/// Read data from a `CubeTensor` synchronously
#[allow(unused, reason = "useful for debugging kernels")]
pub fn into_data_sync<R: CubeRuntime>(tensor: CubeTensor<R>) -> TensorData {
    burn_std::future::block_on(into_data(tensor)).unwrap()
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(tensor, device))
)]
pub(crate) fn to_device<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    device: &R::Device,
) -> CubeTensor<R> {
    if &tensor.device == device {
        return tensor;
    }

    let tensor = kernel::into_contiguous_aligned(tensor);
    let client = R::client(device);
    tensor.to_client(client, device.clone())
}

pub(crate) fn empty<R: CubeRuntime>(
    shape: Shape,
    device: &R::Device,
    dtype: DType,
) -> CubeTensor<R> {
    let client = R::client(device);
    let buffer = client.empty(shape.num_elements() * dtype.size());

    CubeTensor::new_contiguous(client, device.clone(), shape, buffer, dtype)
}

pub(crate) fn swap_dims<R: CubeRuntime>(
    mut tensor: CubeTensor<R>,
    dim1: usize,
    dim2: usize,
) -> CubeTensor<R> {
    tensor.strides.swap(dim1, dim2);
    tensor.shape = tensor.shape.swap(dim1, dim2).unwrap();

    if let DType::QFloat(scheme) = tensor.dtype
        && let QuantLevel::Block(block_size) = scheme.level
    {
        let rank = tensor.rank();
        let qparams = tensor.qparams.as_mut().unwrap();
        let mut block_size = block_size.to_dim_vec(rank);
        block_size.swap(dim1, dim2);

        // Truncate unit dims from the start
        let block_size = block_size
            .into_iter()
            .skip_while(|it| *it == 1)
            .collect::<Vec<_>>();
        if block_size.len() > BlockSize::MAX_DIMS {
            panic!("Swapped block size would exceed max dims");
        }

        qparams.scales.shape.dims.swap(dim1, dim2);
        qparams.scales.strides.swap(dim1, dim2);

        tensor.dtype = DType::QFloat(scheme.with_level(QuantLevel::block(&block_size)))
    }

    tensor
}

/// Permute a tensor's dimensions
pub fn permute<R: CubeRuntime>(mut tensor: CubeTensor<R>, axes: &[usize]) -> CubeTensor<R> {
    // remap strides
    tensor.strides = axes.iter().map(|i| tensor.strides[*i]).collect();

    // remap shape
    tensor.shape = tensor.shape.permute(axes).unwrap();

    if let DType::QFloat(scheme) = tensor.dtype
        && let QuantLevel::Block(block_size) = scheme.level
    {
        let rank = tensor.rank();
        let qparams = tensor.qparams.as_mut().unwrap();

        let mut block_size = block_size.to_dim_vec(rank);
        block_size = axes.iter().map(|i| block_size[*i]).collect();

        // Truncate unit dims from the start
        let block_size = block_size
            .into_iter()
            .skip_while(|it| *it == 1)
            .collect::<Vec<_>>();
        if block_size.len() > BlockSize::MAX_DIMS {
            panic!("Swapped block size would exceed max dims");
        }

        qparams.scales.strides = axes.iter().map(|i| qparams.scales.strides[*i]).collect();
        qparams.scales.shape = qparams.scales.shape.clone().permute(axes).unwrap();

        tensor.dtype = DType::QFloat(scheme.with_level(QuantLevel::block(&block_size)))
    }

    tensor
}

/// Permute a tensor's dimensions from NCHW to NHWC, or the N-dimensional equivalent
pub fn permute_nchw_to_nhwc<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    let rank = tensor.shape.num_dims();
    let c_dim = 1;

    let mut dims = vec![0];
    dims.extend(2..rank);
    dims.push(c_dim);

    permute(tensor, &dims)
}

/// Permute a shape's dimensions from NCHW to NHWC, or the N-dimensional equivalent
pub fn permute_nchw_to_nhwc_shape(shape: Shape) -> Shape {
    let rank = shape.num_dims();
    let c_dim = 1;

    let mut dims = vec![0];
    dims.extend(2..rank);
    dims.push(c_dim);

    shape.permute(&dims).expect("Shape permute should succeed")
}

/// Permute a tensor's dimensions from NHWC to NCHW, or the N-dimensional equivalent
pub fn permute_nhwc_to_nchw<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    let rank = tensor.shape.num_dims();
    let c_dim = rank - 1;

    let mut dims = vec![0];
    dims.push(c_dim);
    dims.extend(1..c_dim);

    permute(tensor, &dims)
}

/// Permute a shape's dimensions from NHWC to NCHW, or the N-dimensional equivalent
pub fn permute_nhwc_to_nchw_shape(shape: Shape) -> Shape {
    let rank = shape.num_dims();
    let c_dim = rank - 1;

    let mut dims = vec![0];
    dims.push(c_dim);
    dims.extend(1..c_dim);

    shape.permute(&dims).expect("Shape permute should succeed")
}

/// Convenience wrapper to permute a 5D tensor's dimensions from NCDHW to NDHWC.
/// Internally this delegates to [`permute_nchw_to_nhwc`], which handles the
/// corresponding N-dimensional permutation pattern.
pub fn permute_ncdhw_to_ndhwc<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    // This is the same as permute_nchw_to_nhwc but more explicit for 5D
    permute_nchw_to_nhwc(tensor)
}

/// Convenience wrapper to permute a 5D tensor's dimensions from NDHWC to NCDHW
/// Internally this delegates to [`permute_nhwc_to_nchw`], which handles the corresponding
/// N-dimensional permutation pattern and supports arbitrary ranks, including 5D.
pub fn permute_ndhwc_to_ncdhw<R: CubeRuntime>(tensor: CubeTensor<R>) -> CubeTensor<R> {
    // This is the same as permute_nhwc_to_nchw but more explicit for 5D
    permute_nhwc_to_nchw(tensor)
}

pub(crate) fn expand<R: CubeRuntime>(tensor: CubeTensor<R>, target_shape: Shape) -> CubeTensor<R> {
    let ndims_in = tensor.shape.num_dims();
    let ndims_out = target_shape.num_dims();

    // Initialize new strides with zeros
    let mut new_strides = vec![0usize; ndims_out];

    // Calculate the difference in dimensions
    let dim_diff = ndims_out.saturating_sub(ndims_in);

    // Compare dimensions from the end, setting strides for matching dimensions or broadcasted ones
    let mut tensor_dim_iter = tensor.shape.iter().rev();
    for i in (0..ndims_out).rev() {
        if i >= dim_diff {
            if let Some(&tensor_dim) = tensor_dim_iter.next() {
                if tensor_dim == target_shape[i] || tensor_dim == 1 {
                    // Copy stride for non-broadcast dimensions or set to 0 for broadcast ones
                    new_strides[i] = if tensor_dim == target_shape[i] {
                        tensor.strides[i - dim_diff]
                    } else {
                        0
                    };
                } else {
                    // Error handling: Dimension mismatch for broadcasting
                    panic!(
                        "Dimension mismatch: cannot broadcast dimension {tensor_dim} of tensor to target shape"
                    );
                }
            } else {
                // If the input tensor has fewer dimensions, treat missing dimensions as 1
                // and set stride to 0 (broadcasting)
                new_strides[i] = 0;
            }
        } else {
            // For extra dimensions in the target shape, set stride to 0 (broadcasting)
            new_strides[i] = 0;
        }
    }

    // Extra check to ensure block scales must be properly handled once they're added
    if tensor.qparams.is_some() {
        match tensor.scheme().level {
            QuantLevel::Tensor => {}
            QuantLevel::Block(_) => todo!(),
        }
    }

    CubeTensor {
        client: tensor.client,
        device: tensor.device,
        shape: target_shape,
        strides: new_strides,
        handle: tensor.handle,
        dtype: tensor.dtype,
        qparams: tensor.qparams,
    }
}

/// Reshape a jit tensor to a new shape
pub fn reshape<R: CubeRuntime>(mut tensor: CubeTensor<R>, shape: Shape) -> CubeTensor<R> {
    let analysis = reshape_action(&tensor.shape.dims, &tensor.strides, &shape.dims);

    match analysis {
        ReshapeAction::UpdateStrides { strides } => {
            tensor.shape = shape;
            tensor.strides = strides;
            return tensor;
        }
        ReshapeAction::NoChange => return tensor,
        ReshapeAction::Recompute => (),
    }

    let tensor = kernel::into_contiguous(tensor);

    let mut out = CubeTensor::new_contiguous(
        tensor.client,
        tensor.device,
        shape,
        tensor.handle,
        tensor.dtype,
    );
    out.qparams = tensor.qparams;
    out
}

/// Reshape a jit tensor to a new shape
pub fn q_reshape<R: CubeRuntime>(mut tensor: CubeTensor<R>, shape: Shape) -> CubeTensor<R> {
    let scheme = *tensor.scheme();

    let shape_values = {
        let rank = shape.num_dims();
        let mut shape = shape.clone();
        shape[rank - 1] = shape[rank - 1].div_ceil(scheme.num_quants());
        shape
    };
    let shape_scales = params_shape(&shape, scheme.level);
    let (values, scales) = tensor.quantized_handles().unwrap();

    let analysis_values = reshape_action(&values.shape.dims, &values.strides, &shape_values.dims);
    let analysis_scales = reshape_action(&scales.shape.dims, &scales.strides, &shape_scales.dims);

    match (analysis_values, analysis_scales) {
        (
            ReshapeAction::UpdateStrides { strides },
            ReshapeAction::UpdateStrides {
                strides: scales_strides,
            },
        ) => {
            let qparams = tensor.qparams.as_mut().unwrap();

            tensor.shape = shape;
            tensor.strides = strides;

            qparams.scales.shape = shape_scales;
            qparams.scales.strides = scales_strides;
        }
        (ReshapeAction::UpdateStrides { strides }, ReshapeAction::NoChange) => {
            tensor.shape = shape;
            tensor.strides = strides;
        }
        (
            ReshapeAction::NoChange,
            ReshapeAction::UpdateStrides {
                strides: scales_strides,
            },
        ) => {
            let qparams = tensor.qparams.as_mut().unwrap();

            qparams.scales.shape = shape_scales;
            qparams.scales.strides = scales_strides;
        }
        (ReshapeAction::NoChange, ReshapeAction::NoChange) => {}
        _ => {
            tensor = kernel::into_contiguous(tensor);
            tensor.shape = shape;
            tensor.strides = contiguous_strides(&shape_values.dims);

            let qparams = tensor.qparams.as_mut().unwrap();

            qparams.scales.strides = contiguous_strides(&shape_scales.dims);
            qparams.scales.shape = shape_scales;
        }
    }

    tensor
}

pub(crate) fn max_line_size<R: CubeRuntime>(tensor: &CubeTensor<R>) -> LineSize {
    tensor_line_size_parallel(
        tensor
            .client
            .io_optimized_line_sizes_unchecked(tensor.dtype.size()),
        &tensor.shape,
        &tensor.strides,
        tensor.shape.len() - 1,
    )
}

pub(crate) fn max_line_size_many<R: CubeRuntime>(
    tensors: &[&CubeTensor<R>],
    axis: usize,
) -> LineSize {
    let vec = tensors
        .iter()
        .map(|tensor| {
            tensor_line_size_parallel(
                tensor
                    .client
                    .io_optimized_line_sizes_unchecked(tensor.dtype.size()),
                &tensor.shape,
                &tensor.strides,
                axis,
            )
        })
        .min();

    vec.unwrap_or(0)
}

/// Unfold windows along a dimension.
///
/// Returns a view of the tensor with all complete windows of size `size` in dimension `dim`;
/// where windows are advanced by `step` at each index.
///
/// The number of windows is `max(0, (shape[dim] - size).ceil_div(step))`.
///
/// The new view will have the unfolded dimension replaced by two dimensions;
/// one in the position of the original dimension, with size equal to the number of windows,
/// and one appended to the right-most position, with size equal to `size`.
///
/// # Arguments
///
/// * `tensor` - The input tensor to unfold; of shape ``[pre=..., dim shape, post=...]``
/// * `dim` - the dimension to unfold.
/// * `size` - the size of each unfolded window.
/// * `step` - the step between each window.
///
/// # Returns
///
/// A tensor view with the shape ``[pre=..., windows, post=..., size]``.
pub fn unfold<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    dim: usize,
    size: usize,
    step: usize,
) -> CubeTensor<R> {
    let shape = calculate_unfold_shape(tensor.shape, dim, size, step);

    let d_stride = tensor.strides[dim];
    let mut strides = tensor.strides.clone();
    strides[dim] = step * d_stride;
    strides.push(d_stride);

    CubeTensor {
        shape,
        strides,
        ..tensor
    }
}
