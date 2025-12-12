use serde::{Deserialize, Serialize};

use burn_backend::{DType, Shape};

/// The tensor unique identifier.
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Debug, Serialize, Deserialize)]
pub struct TensorId {
    value: u64,
}

impl core::fmt::Display for TensorId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("TensorId({:?})", self.value))
    }
}

/// The status of the current tensor.
#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorStatus {
    /// The tensor can be read, but not written.
    ReadOnly,
    /// The tensor can be mutated inplace.
    ReadWrite,
    /// No handle exists for that tensor.
    NotInit,
}

/// A tensor definition represents a snapshot of a tensor when it was used.
///
/// # Example
///
/// A tensor that is used multiple times has its status updated for each operation.
///
///   1. Status::NotInit
///   2. Status::ReadOnly
///   3. Status::ReadOnly
///   4. Status::ReadWrite
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorIr {
    /// The [tensor id](TensorId).
    pub id: TensorId,
    /// The shape of the tensor.
    pub shape: Shape,
    /// The [status](TensorStatus) of the tensor when it was used.
    pub status: TensorStatus,
    /// The [type](DType) of the tensor.
    pub dtype: DType,
}

impl TensorId {
    /// Create a new tensor id.
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}

impl TensorIr {
    /// Create a new tensor that is not already initialized.
    pub fn uninit(id: TensorId, shape: Shape, dtype: DType) -> Self {
        Self {
            id,
            status: TensorStatus::NotInit,
            shape,
            dtype,
        }
    }
}
