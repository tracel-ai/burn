use crate::TensorSnapshot;
use burn_core::module::ParamId;
use burn_tensor::{DType, TensorData};

/// Helper to create a test TensorSnapshot
#[allow(dead_code)]
pub fn create_test_snapshot(
    name: String,
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: DType,
) -> TensorSnapshot {
    TensorSnapshot::from_data(
        TensorData::from_bytes_vec(data, shape, dtype),
        vec![name],
        vec![],
        ParamId::new(),
    )
}
