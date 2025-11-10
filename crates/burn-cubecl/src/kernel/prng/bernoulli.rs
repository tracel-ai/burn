use burn_tensor::Shape;
use cubecl::prelude::*;

use crate::{CubeRuntime, element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor};

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli<R: CubeRuntime, E: CubeElement + Numeric>(
    shape: Shape,
    device: &R::Device,
    probability: f32,
) -> CubeTensor<R> {
    let client = R::client(device);
    let output = empty_device::<R, E>(client.clone(), device.clone(), shape);

    cubecl::random::random_bernoulli::<R>(
        &client,
        probability,
        output.as_handle_ref(),
        E::dtype().into(),
    );

    output
}
