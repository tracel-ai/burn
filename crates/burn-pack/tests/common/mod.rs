//! Shared helpers for the burn-pack format integration tests.
//!
//! Included by multiple test binaries; not every helper is used by every one.
#![allow(dead_code)]

use burn_pack::{Bytes, DType, Tensor};

/// Build an f32 [`Tensor`] entry from values + shape.
pub fn f32_tensor(name: &str, values: &[f32], shape: &[usize], param_id: Option<u64>) -> Tensor {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Tensor::new(
        name.to_string(),
        DType::F32,
        shape.to_vec(),
        param_id,
        Bytes::from_bytes_vec(raw),
    )
}

/// Build a [`Tensor`] entry from raw little-endian bytes with an explicit dtype.
pub fn raw_tensor(
    name: &str,
    dtype: DType,
    shape: &[usize],
    bytes: Vec<u8>,
    param_id: Option<u64>,
) -> Tensor {
    Tensor::new(
        name.to_string(),
        dtype,
        shape.to_vec(),
        param_id,
        Bytes::from_bytes_vec(bytes),
    )
}

/// Decode a tensor entry's bytes as f32 values.
pub fn read_f32(tensor: &Tensor) -> Vec<f32> {
    let slice: &[u8] = &tensor.bytes;
    slice
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
