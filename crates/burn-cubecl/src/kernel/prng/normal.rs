use crate::{CubeDevice, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::RandomKey;
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape};

/// Pseudo-random generator with uniform distribution
pub fn random_normal(
    shape: Shape,
    device: &CubeDevice,
    mean: f32,
    std: f32,
    dtype: DType,
    key: RandomKey,
) -> CubeTensor {
    let client = device.client();
    let output = empty_device_dtype(client.clone(), device.clone(), shape, dtype);

    cubek::random::random_normal(
        &client,
        mean,
        std,
        output.clone().binding(),
        dtype_to_storage_type(dtype),
        key.seed(),
    )
    .expect("Kernel to never fail");

    output
}
