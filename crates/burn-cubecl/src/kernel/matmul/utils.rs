use burn_tensor::calculate_matmul_output;

use crate::{CubeRuntime, element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor};

/// Creates an empty output tensor with matmul output shape
pub fn init_matmul_output<R: CubeRuntime, E: CubeElement>(
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
) -> CubeTensor<R> {
    empty_device::<R, E>(
        lhs.client.clone(),
        lhs.device.clone(),
        calculate_matmul_output(&lhs.shape, &rhs.shape).unwrap(),
    )
}
