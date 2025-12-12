use crate::{CubeRuntime, ops::numeric::empty_device_optimized_dtype, tensor::CubeTensor};
use burn_backend::{DType, calculate_matmul_output};

/// Creates an empty output tensor with matmul output shape
pub fn init_matmul_output<R: CubeRuntime>(
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
    dtype: DType,
) -> CubeTensor<R> {
    empty_device_optimized_dtype(
        lhs.client.clone(),
        lhs.device.clone(),
        calculate_matmul_output(&lhs.shape, &rhs.shape).unwrap(),
        dtype,
    )
}
