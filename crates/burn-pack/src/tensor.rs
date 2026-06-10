//! Tensor-library-agnostic tensor entry for the burnpack format.
//!
//! The burnpack reader and writer operate on [`Tensor`] values, which
//! carry only the format-level information (name, dtype, shape, optional param
//! id) plus a lazy provider of the raw little-endian bytes. This keeps the
//! format crate free of any tensor library dependency: higher layers (e.g.
//! `burn-core`) bridge between [`Tensor`] and their own tensor/snapshot
//! types.

use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;

use burn_std::Bytes;
use burn_std::DType;

use super::base::Error;

/// A lazy provider of a tensor's raw little-endian bytes.
pub type TensorBytesFn = Rc<dyn Fn() -> Result<Bytes, Error>>;

/// A single tensor in a burnpack container, decoupled from any tensor library.
///
/// Holds the metadata needed to (de)serialize the tensor plus a closure that
/// materializes its raw bytes on demand. The byte buffer is the tensor's data
/// in little-endian layout, matching `byte_len` and the element count implied
/// by `shape` and `dtype`.
#[derive(Clone)]
pub struct Tensor {
    /// Fully-qualified tensor name (e.g. `"encoder.layer1.weight"`).
    pub name: String,
    /// Data type of the tensor.
    pub dtype: DType,
    /// Tensor shape dimensions.
    pub shape: Vec<usize>,
    /// Optional parameter id, used to preserve identities for stateful training.
    pub param_id: Option<u64>,
    /// Number of raw bytes the tensor occupies.
    pub byte_len: usize,
    /// Lazy provider of the raw bytes.
    data: TensorBytesFn,
}

impl Tensor {
    /// Create a tensor entry from its metadata and a lazy byte provider.
    pub fn new(
        name: String,
        dtype: DType,
        shape: Vec<usize>,
        param_id: Option<u64>,
        byte_len: usize,
        data: TensorBytesFn,
    ) -> Self {
        Self {
            name,
            dtype,
            shape,
            param_id,
            byte_len,
            data,
        }
    }

    /// Create a tensor entry from already-materialized bytes.
    pub fn from_bytes(
        name: String,
        dtype: DType,
        shape: Vec<usize>,
        param_id: Option<u64>,
        bytes: Bytes,
    ) -> Self {
        let byte_len = bytes.len();
        let data: TensorBytesFn = Rc::new(move || Ok(bytes.clone()));
        Self {
            name,
            dtype,
            shape,
            param_id,
            byte_len,
            data,
        }
    }

    /// Materialize the tensor's raw bytes.
    pub fn bytes(&self) -> Result<Bytes, Error> {
        (self.data)()
    }
}
