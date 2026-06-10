//! Tensor-library-agnostic tensor entry for the burnpack format.
//!
//! The burnpack reader and writer operate on [`Tensor`] values, which carry only the
//! format-level information (name, dtype, shape, optional param id) plus the raw
//! little-endian [`Bytes`]. Keeping the bytes as [`Bytes`] (rather than a custom buffer
//! type) integrates with the rest of the Burn ecosystem: a reader can hand out
//! file-backed bytes ([`Bytes::from_file`]) for fast, lazy file-to-GPU transfers, while a
//! writer simply consumes already-materialized bytes.

use alloc::string::String;

use burn_std::{Bytes, DType, Shape};

/// A single tensor in a burnpack container, decoupled from any tensor library.
///
/// The [`bytes`](Self::bytes) field holds the tensor's data in little-endian layout,
/// matching the element count implied by [`shape`](Self::shape) and [`dtype`](Self::dtype).
/// When produced by a [`Reader`](crate::Reader) loading from a file, the bytes are
/// file-backed and only read from disk when accessed.
#[derive(Clone)]
pub struct Tensor {
    /// Fully-qualified tensor name (e.g. `"encoder.layer1.weight"`).
    pub name: String,
    /// Data type of the tensor.
    pub dtype: DType,
    /// Tensor shape.
    pub shape: Shape,
    /// Optional parameter id, used to preserve identities for stateful training.
    pub param_id: Option<u64>,
    /// The tensor's raw little-endian bytes.
    pub bytes: Bytes,
}

impl Tensor {
    /// Create a tensor entry from its metadata and raw bytes.
    pub fn new(
        name: String,
        dtype: DType,
        shape: impl Into<Shape>,
        param_id: Option<u64>,
        bytes: Bytes,
    ) -> Self {
        Self {
            name,
            dtype,
            shape: shape.into(),
            param_id,
            bytes,
        }
    }

    /// Number of raw bytes the tensor occupies.
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}
