use crate::{CubeDevice, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape, TensorMetadata};

/// Pseudo-random generator with uniform distribution
pub fn random_uniform(
    shape: Shape,
    device: &CubeDevice,
    lower_bound: f32,
    upper_bound: f32,
    dtype: DType,
) -> CubeTensor {
    let client = device.client();
    let output = empty_device_dtype(client.clone(), device.clone(), shape, dtype);

    cubek::random::random_uniform(
        &client,
        lower_bound,
        upper_bound,
        output.clone().binding(),
        dtype_to_storage_type(dtype),
    )
    .expect("Kernel to never fail");

    output
}

/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_uniform(
    tensor: &CubeTensor,
    lower_bound: f32,
    upper_bound: f32,
    dtype: DType,
) -> CubeTensor {
    random_uniform(
        tensor.shape(),
        &tensor.device,
        lower_bound,
        upper_bound,
        dtype,
    )
}
