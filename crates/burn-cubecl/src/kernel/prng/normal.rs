use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{DType, Shape};

/// Pseudo-random generator with uniform distribution
pub fn random_normal<R: CubeRuntime>(
    shape: Shape,
    device: &R::Device,
    mean: f32,
    std: f32,
    dtype: DType,
) -> CubeTensor<R> {
    let client = R::client(device);
    let output = empty_device_dtype(client.clone(), device.clone(), shape, dtype);

    cubek::random::random_normal(&client, mean, std, output.clone().binding(), dtype.into())
        .expect("Kernel to never fail");

    output
}
