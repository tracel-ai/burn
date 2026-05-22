use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{DType, Shape};

/// Pseudo-random generator with bernoulli distribution
pub fn random_bernoulli<R: CubeRuntime>(
    shape: Shape,
    device: &R::Device,
    probability: f32,
    dtype: DType,
) -> CubeTensor<R> {
    let client = R::client(device);
    let output = empty_device_dtype(client.clone(), device.clone(), shape, dtype);

    cubek::random::random_bernoulli(
        &client,
        probability,
        output.clone().binding(),
        dtype_to_storage_type(dtype),
    )
    .expect("Kernel to never fail");

    output
}
