use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_tensor::{DType, Shape};

/// Pseudo-random generator with uniform distribution
pub fn random_uniform<R: CubeRuntime>(
    shape: Shape,
    device: &R::Device,
    lower_bound: f32,
    upper_bound: f32,
    dtype: DType,
) -> CubeTensor<R> {
    let client = R::client(device);
    let output = empty_device_dtype(client.clone(), device.clone(), shape, dtype);
    let output_handle = output.as_handle_ref();

    cubecl::random::random_uniform(
        &client,
        lower_bound,
        upper_bound,
        output_handle,
        dtype.into(),
    );

    output
}

/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_uniform<R: CubeRuntime>(
    tensor: &CubeTensor<R>,
    lower_bound: f32,
    upper_bound: f32,
    dtype: DType,
) -> CubeTensor<R> {
    random_uniform(
        tensor.shape.clone(),
        &tensor.device,
        lower_bound,
        upper_bound,
        dtype,
    )
}
