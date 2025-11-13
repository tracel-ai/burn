use burn_tensor::Shape;

use crate::{CubeRuntime, element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor};

/// Pseudo-random generator with uniform distribution
pub fn random_uniform<R: CubeRuntime, E: CubeElement>(
    shape: Shape,
    device: &R::Device,
    lower_bound: E,
    upper_bound: E,
) -> CubeTensor<R> {
    let client = R::client(device);
    let output = empty_device::<R, E>(client.clone(), device.clone(), shape);
    let output_handle = output.as_handle_ref();

    cubecl::random::random_uniform(
        &client,
        lower_bound.elem::<f32>(),
        upper_bound.elem::<f32>(),
        output_handle,
        E::dtype().into(),
    );

    output
}

/// Pseudo-random generator for uniform distribution, based on
/// another tensor.
pub fn random_like_uniform<R: CubeRuntime, E: CubeElement>(
    tensor: &CubeTensor<R>,
    lower_bound: E,
    upper_bound: E,
) -> CubeTensor<R> {
    random_uniform(
        tensor.shape.clone(),
        &tensor.device,
        lower_bound,
        upper_bound,
    )
}
