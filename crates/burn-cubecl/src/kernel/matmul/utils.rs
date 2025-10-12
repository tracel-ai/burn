use crate::{CubeRuntime, element::CubeElement, ops::numeric::empty_device, tensor::CubeTensor};
use burn_tensor::Shape;

/// Creates an empty output tensor with matmul output shape
pub fn init_matmul_output<R: CubeRuntime, E: CubeElement>(
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
) -> CubeTensor<R> {
    empty_device::<R, E>(
        lhs.client.clone(),
        lhs.device.clone(),
        Shape::matmul(&lhs.shape, &rhs.shape).unwrap(),
    )
}
