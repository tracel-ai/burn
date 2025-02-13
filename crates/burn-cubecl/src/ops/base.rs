use crate::{element::CubeElement, kernel, tensor::CubeTensor, CubeRuntime};
use burn_tensor::{Shape, TensorData};
use cubecl::tensor_vectorization_factor;

pub(crate) fn from_data<R: CubeRuntime>(data: TensorData, device: &R::Device) -> CubeTensor<R> {
    let shape: Shape = (&data.shape).into();
    let client = R::client(device);
    let buffer = client.create(data.as_bytes());

    CubeTensor::new_contiguous(client, device.clone(), shape, buffer, data.dtype)
}

pub(crate) async fn into_data<R: CubeRuntime, E: CubeElement>(tensor: CubeTensor<R>) -> TensorData {
    let tensor = kernel::into_contiguous(tensor);

    let bytes = tensor.client.read_one_async(tensor.handle.binding()).await;
    let actual_len = tensor.shape.num_elements() * size_of::<E>();
    TensorData::new(E::from_bytes(&bytes[..actual_len]).to_vec(), tensor.shape)
}

/// Read data from a `CubeTensor` synchronously
#[allow(unused, reason = "useful for debugging kernels")]
pub fn into_data_sync<R: CubeRuntime, E: CubeElement>(tensor: CubeTensor<R>) -> TensorData {
    let tensor = kernel::into_contiguous(tensor);

    let bytes = tensor.client.read_one(tensor.handle.binding());
    let actual_len = tensor.shape.num_elements() * size_of::<E>();
    TensorData::new(E::from_bytes(&bytes[..actual_len]).to_vec(), tensor.shape)
}

pub(crate) fn to_device<R: CubeRuntime>(
    tensor: CubeTensor<R>,
    device: &R::Device,
) -> CubeTensor<R> {
    if &tensor.device == device {
        return tensor;
    }

    let client = R::client(device);
    tensor.to_client(client, device.clone())
}

pub(crate) fn empty<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
) -> CubeTensor<R> {
    let client = R::client(device);
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    CubeTensor::new_contiguous(client, device.clone(), shape, buffer, E::dtype())
}

pub(crate) fn swap_dims<R: CubeRuntime>(
    mut tensor: CubeTensor<R>,
    dim1: usize,
    dim2: usize,
) -> CubeTensor<R> {
    tensor.strides.swap(dim1, dim2);
    tensor.shape.dims.swap(dim1, dim2);

    tensor
}

/// Permute a tensor's dimensions
pub fn permute<R: CubeRuntime>(mut tensor: CubeTensor<R>, axes: &[usize]) -> CubeTensor<R> {
    // remap strides
    tensor.strides = axes.iter().map(|i| tensor.strides[*i]).collect();

    // remap shape
    tensor.shape.dims = axes.iter().map(|i| tensor.shape.dims[*i]).collect();

    tensor
}
pub(crate) fn expand<R: CubeRuntime>(tensor: CubeTensor<R>, target_shape: Shape) -> CubeTensor<R> {
    let ndims_in = tensor.shape.num_dims();
    let ndims_out = target_shape.num_dims();

    // Initialize new strides with zeros
    let mut new_strides = vec![0usize; ndims_out];

    // Calculate the difference in dimensions
    let dim_diff = ndims_out.saturating_sub(ndims_in);

    // Compare dimensions from the end, setting strides for matching dimensions or broadcasted ones
    let mut tensor_dim_iter = tensor.shape.dims.iter().rev();
    for i in (0..ndims_out).rev() {
        if i >= dim_diff {
            if let Some(&tensor_dim) = tensor_dim_iter.next() {
                if tensor_dim == target_shape.dims[i] || tensor_dim == 1 {
                    // Copy stride for non-broadcast dimensions or set to 0 for broadcast ones
                    new_strides[i] = if tensor_dim == target_shape.dims[i] {
                        tensor.strides[i - dim_diff]
                    } else {
                        0
                    };
                } else {
                    // Error handling: Dimension mismatch for broadcasting
                    panic!(
                        "Dimension mismatch: cannot broadcast dimension {} of tensor to target shape",
                        tensor_dim
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

    CubeTensor {
        client: tensor.client,
        device: tensor.device,
        shape: target_shape,
        strides: new_strides,
        handle: tensor.handle,
        dtype: tensor.dtype,
    }
}

/// Reshape a jit tensor to a new shape
pub fn reshape<R: CubeRuntime>(tensor: CubeTensor<R>, shape: Shape) -> CubeTensor<R> {
    // TODO: Not force standard layout all the time (improve performance).
    let tensor = kernel::into_contiguous(tensor);

    CubeTensor::new_contiguous(
        tensor.client,
        tensor.device,
        shape,
        tensor.handle,
        tensor.dtype,
    )
}

pub(crate) fn max_vectorization<R: CubeRuntime>(tensor: &CubeTensor<R>) -> u8 {
    tensor_vectorization_factor(
        R::supported_line_sizes(),
        &tensor.shape.dims,
        &tensor.strides,
        tensor.shape.num_dims() - 1,
    )
}
