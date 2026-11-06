use crate::{CubeDevice, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::RandomKey;
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape};

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli(
    shape: Shape,
    device: &CubeDevice,
    probability: f32,
    dtype: DType,
    key: RandomKey,
) -> CubeTensor {
    let client = device.client();
    let output = empty_device_dtype(client.clone(), device.clone(), shape, dtype);

    cubek::random::random_bernoulli(
        &client,
        probability,
        output.clone().binding(),
        dtype_to_storage_type(dtype),
        key.seed(),
    )
    .expect("Kernel to never fail");

    output
}
