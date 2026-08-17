//! Tensor-library-agnostic tensor entry for the burnpack format.
//!
//! [`Tensor`] carries only the format-level information (name, dtype, shape, optional param
//! id) plus the raw little-endian [`Bytes`]. Keeping the bytes as [`Bytes`] (rather than a
//! custom buffer type) integrates with the rest of the Burn ecosystem: a reader can hand out
//! file-backed bytes ([`Bytes::from_file`]) for fast, lazy file-to-GPU transfers.
//!
//! A tensor's bytes need not exist yet. [`Tensor::deferred`] describes one by its length and
//! a provider that yields the data on demand, which is what lets [`Writer`](crate::Writer)
//! save a model without holding all of it in memory.

use alloc::string::String;
use alloc::sync::Arc;

use burn_std::{Bytes, DType, Shape};

use crate::base::Error;

/// Produces a deferred tensor's bytes when the writer reaches it.
///
/// `Send + Sync` because a [`Tensor`] must stay so: records holding one cross threads through
/// burn-train's async checkpointer. `Arc` rather than `Box<dyn FnOnce>` so [`Tensor`] stays
/// [`Clone`], which the reader-facing API relies on.
type Provider = Arc<dyn Fn() -> Result<Bytes, Error> + Send + Sync>;

/// Where a tensor's bytes come from.
///
/// The distinction is invisible to readers of the container: both variants describe the same
/// `byte_len` bytes at the same offset. It matters only to the writer, which plans the whole
/// container from lengths alone and then draws the data one tensor at a time.
#[derive(Clone)]
enum Source {
    /// Bytes already in hand.
    Resident(Bytes),
    /// Bytes not yet produced, but whose length is already known.
    Deferred { len: usize, provider: Provider },
}

/// A single tensor in a burnpack container, decoupled from any tensor library.
///
/// The bytes are in little-endian layout, matching the element count implied by
/// [`shape`](Self::shape) and [`dtype`](Self::dtype), and are reached with
/// [`into_bytes`](Self::into_bytes). Their length is available up front via
/// [`byte_len`](Self::byte_len) whether or not they exist yet.
///
/// Construct one with [`new`](Self::new) when the data is already in memory, or
/// [`deferred`](Self::deferred) when it is not. A [`Reader`](crate::Reader) produces resident
/// tensors, though for a file-backed source the bytes are still only read from disk on
/// access.
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
    /// Where the raw little-endian bytes come from.
    source: Source,
}

impl Tensor {
    /// Create a tensor from its metadata and raw bytes.
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
            source: Source::Resident(bytes),
        }
    }

    /// Create a tensor whose bytes are produced on demand.
    ///
    /// `byte_len` must equal the length of the [`Bytes`] `provider` eventually returns, since
    /// [`Writer`](crate::Writer) reserves exactly that much space in the data section before
    /// calling it. Two checks enforce it, because getting it wrong would misplace every later
    /// tensor: one while planning, against `shape` and `dtype` (skipped for quantized data,
    /// whose length is not a product of the two), and one while writing, against the bytes
    /// actually produced.
    ///
    /// `provider` runs at most once per write, when the writer reaches this tensor, and its
    /// result is dropped before the next tensor's is requested. Producing the data there -
    /// reading a parameter back from a device, say - is what bounds a save's peak host memory
    /// by the largest single tensor rather than by the whole model.
    pub fn deferred(
        name: String,
        dtype: DType,
        shape: impl Into<Shape>,
        param_id: Option<u64>,
        byte_len: usize,
        provider: impl Fn() -> Result<Bytes, Error> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name,
            dtype,
            shape: shape.into(),
            param_id,
            source: Source::Deferred {
                len: byte_len,
                provider: Arc::new(provider),
            },
        }
    }

    /// Number of raw bytes the tensor occupies, known without producing them.
    pub fn byte_len(&self) -> usize {
        match &self.source {
            Source::Resident(bytes) => bytes.len(),
            Source::Deferred { len, .. } => *len,
        }
    }

    /// Whether the bytes are already in memory, as opposed to produced on demand.
    pub fn is_resident(&self) -> bool {
        matches!(self.source, Source::Resident(_))
    }

    /// Take the tensor's raw little-endian bytes, producing them if deferred.
    ///
    /// Infallible in practice for a tensor from a [`Reader`](crate::Reader), which only
    /// produces resident tensors; the [`Result`] is there for deferred ones, whose provider
    /// can fail.
    pub fn into_bytes(self) -> Result<Bytes, Error> {
        match self.source {
            Source::Resident(bytes) => Ok(bytes),
            Source::Deferred { provider, .. } => provider(),
        }
    }
}
