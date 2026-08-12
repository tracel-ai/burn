//! Tensor-library-agnostic tensor entry for the burnpack format.
//!
//! The burnpack reader produces [`Tensor`] values, which carry only the format-level
//! information (name, dtype, shape, optional param id) plus the raw little-endian
//! [`Bytes`]. Keeping the bytes as [`Bytes`] (rather than a custom buffer type)
//! integrates with the rest of the Burn ecosystem: a reader can hand out file-backed
//! bytes ([`Bytes::from_file`]) for fast, lazy file-to-GPU transfers.
//!
//! The writer is more permissive: it accepts anything implementing [`TensorEntry`], which
//! [`Tensor`] implements for bytes that are already resident.

use alloc::borrow::Cow;
use alloc::string::String;

use burn_std::{Bytes, DType, Shape};

use crate::base::Error;

/// A tensor the [`Writer`](crate::Writer) can emit: format-level metadata plus a source of
/// the raw little-endian bytes that is only drawn from when the writer reaches that tensor.
///
/// The writer plans the entire container - descriptors, offsets, total size - from
/// [`byte_len`](Self::byte_len) alone, before any I/O happens. Only then does it call
/// [`into_bytes`](Self::into_bytes), once per tensor, in write order, dropping each
/// tensor's bytes before asking for the next.
///
/// An implementation that defers materialization until `into_bytes` therefore holds only
/// one tensor's data at a time. Paired with [`Writer::write_to_file`](crate::Writer::write_to_file),
/// which streams straight to disk, that bounds peak host memory by the largest single
/// tensor rather than by the whole tensor set. The in-memory sinks
/// ([`into_bytes`](crate::Writer::into_bytes), [`write_into`](crate::Writer::write_into))
/// still hold the full container, so they gain less.
pub trait TensorEntry {
    /// Fully-qualified tensor name (e.g. `"encoder.layer1.weight"`).
    ///
    /// Returns [`Cow`] so implementations that store the name can lend it while those that
    /// compute it (by joining a module path, say) can hand over the computed [`String`].
    fn name(&self) -> Cow<'_, str>;

    /// Data type of the tensor.
    fn dtype(&self) -> DType;

    /// Tensor shape.
    fn shape(&self) -> &Shape;

    /// Optional parameter id, used to preserve identities for stateful training.
    fn param_id(&self) -> Option<u64>;

    /// Number of raw bytes the tensor occupies, known without materializing the data.
    ///
    /// Must equal the length of the [`Bytes`] returned by [`into_bytes`](Self::into_bytes).
    /// The writer reserves exactly this much space in the data section and fails with
    /// [`Error::TensorBytesSizeMismatch`] if the two disagree.
    fn byte_len(&self) -> usize;

    /// Produce the tensor's raw little-endian bytes.
    ///
    /// Called once, when the writer reaches this tensor in the data section.
    fn into_bytes(self) -> Result<Bytes, Error>;
}

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

impl TensorEntry for Tensor {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn param_id(&self) -> Option<u64> {
        self.param_id
    }

    fn byte_len(&self) -> usize {
        Tensor::byte_len(self)
    }

    fn into_bytes(self) -> Result<Bytes, Error> {
        Ok(self.bytes)
    }
}
