use crate::{ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{DType, calculate_matmul_output};

/// Creates an empty output tensor with matmul output shape
pub fn init_matmul_output(lhs: &CubeTensor, rhs: &CubeTensor, dtype: DType) -> CubeTensor {
    empty_device_dtype(
        lhs.client.clone(),
        lhs.device.clone(),
        calculate_matmul_output(lhs.meta.shape(), rhs.meta.shape()).unwrap(),
        dtype,
    )
}
