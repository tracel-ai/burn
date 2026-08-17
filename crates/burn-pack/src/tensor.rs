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

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;
// `alloc::sync` needs atomic CAS. A target without it has no threads to share a tensor
// across, so neither the atomic pointer nor the `Send + Sync` bound on [`ByteSource`] is
// needed there, and both are dropped below.
#[cfg(not(target_has_atomic = "ptr"))]
use alloc::rc::Rc as Arc;

use burn_std::{Bytes, DType, Shape};

use crate::base::Error;

/// What a deferred tensor's byte provider must satisfy.
///
/// `Send` on targets with threads, because a [`Tensor`] must stay `Send`: an optimizer record
/// holds a `Vec` of them and crosses threads through burn-train's async checkpointer, whose
/// `Checkpoint` bound is `Send`. `Sync` follows from sharing the provider in an `Arc`
/// (`Arc<T>: Send` requires `T: Send + Sync`), and that sharing is what keeps [`Tensor`]
/// [`Clone`]. A target without atomic CAS has no threads and no `alloc::sync`, so neither
/// bound applies or can be met there.
#[cfg(target_has_atomic = "ptr")]
pub trait ByteSource: Fn() -> Result<Bytes, Error> + Send + Sync + 'static {}
#[cfg(target_has_atomic = "ptr")]
impl<T> ByteSource for T where T: Fn() -> Result<Bytes, Error> + Send + Sync + 'static {}

/// Without atomic CAS there are no threads, so the bound is just the closure itself.
#[cfg(not(target_has_atomic = "ptr"))]
pub trait ByteSource: Fn() -> Result<Bytes, Error> + 'static {}
#[cfg(not(target_has_atomic = "ptr"))]
impl<T> ByteSource for T where T: Fn() -> Result<Bytes, Error> + 'static {}

/// Shared handle to a [`ByteSource`], kept behind a pointer so [`Tensor`] stays [`Clone`].
#[cfg(target_has_atomic = "ptr")]
type Provider = Arc<dyn Fn() -> Result<Bytes, Error> + Send + Sync>;
#[cfg(not(target_has_atomic = "ptr"))]
type Provider = Arc<dyn Fn() -> Result<Bytes, Error>>;

/// Where a tensor's bytes come from.
///
/// The distinction is invisible to readers of the container: both variants describe the same
/// `byte_len` bytes at the same offset. It matters only to the writer, which plans the whole
/// container from lengths alone and then draws the data one tensor at a time.
///
/// [`Bytes`] is itself already lazy about the *copy* - a file-backed or device-backed buffer
/// reports its length without materializing and only reads on access - so `Resident` bytes
/// are not necessarily in host memory. `Deferred` defers a step earlier: the call that would
/// produce the `Bytes` at all. Converting a whole model eagerly would submit every device
/// readback up front even if each returned a lazy buffer, which is what this avoids.
#[derive(Clone)]
enum Source {
    /// Bytes already in hand.
    Resident(Bytes),
    /// Bytes not yet produced, but whose length is already known.
    Deferred { len: usize, provider: Provider },
}

/// A single tensor in a burnpack container, decoupled from any tensor library.
///
/// The bytes are in little-endian layout, and are reached with [`bytes`](Self::bytes) or
/// [`into_bytes`](Self::into_bytes). Their length is available up front via
/// [`byte_len`](Self::byte_len) whether or not they exist yet. For every dtype but
/// [`DType::QFloat`] that length is the element count implied by [`shape`](Self::shape) and
/// [`dtype`](Self::dtype); quantized data packs its values and appends scales inline, so it
/// is not a product of the two.
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
    /// calling it. Getting it wrong would misplace every later tensor, so it is checked twice:
    /// while planning, against `shape` and `dtype`, and while writing, against the bytes
    /// actually produced. Quantized data is exempt from the first, its length being packed
    /// values plus inline scales rather than a product of the two, so there the write-time
    /// check is the only guard.
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
        provider: impl ByteSource,
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

    /// The tensor's raw little-endian bytes, producing them if deferred.
    ///
    /// Leaves the tensor intact, so the metadata stays reachable afterwards. Prefer
    /// [`into_bytes`](Self::into_bytes) wherever the tensor is no longer needed: this method
    /// is cheap only for the in-memory buffers a [`Reader`](crate::Reader) hands out, whose
    /// clone is a refcount bump. Two cases cost the full tensor:
    ///
    /// - Nothing is cached, so a `Deferred` tensor runs its provider again on every call.
    /// - A `Resident` tensor whose [`Bytes`] cannot share its backing is deep-copied. That
    ///   covers a plain heap buffer, and also a device-backed one, where the copy forces the
    ///   whole device-to-host readback the writer otherwise streams in bounded chunks.
    pub fn bytes(&self) -> Result<Bytes, Error> {
        match &self.source {
            Source::Resident(bytes) => Ok(bytes.clone()),
            Source::Deferred { provider, .. } => provider(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    fn resident() -> Tensor {
        Tensor::new(
            "w".to_string(),
            DType::U8,
            vec![4],
            None,
            Bytes::from_bytes_vec(vec![1, 2, 3, 4]),
        )
    }

    /// `bytes` borrows rather than consuming, so everything about the tensor stays reachable
    /// afterwards. That is the whole reason it exists next to `into_bytes`.
    #[test]
    fn bytes_leaves_a_resident_tensor_intact() {
        let tensor = resident();

        assert_eq!(&tensor.bytes().unwrap()[..], &[1, 2, 3, 4]);
        assert_eq!(&tensor.bytes().unwrap()[..], &[1, 2, 3, 4]);
        assert_eq!(tensor.byte_len(), 4);
        assert_eq!(tensor.name, "w");
        assert_eq!(&tensor.into_bytes().unwrap()[..], &[1, 2, 3, 4]);
    }

    /// Nothing is cached, so each call runs the provider again. Documented on `bytes`, and
    /// the reason the record load paths use `into_bytes` instead.
    #[test]
    fn bytes_reruns_a_deferred_provider_on_every_call() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();

        let tensor = Tensor::deferred("w".to_string(), DType::U8, vec![2], None, 2, move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(Bytes::from_bytes_vec(vec![7, 8]))
        });

        assert_eq!(
            calls.load(Ordering::Relaxed),
            0,
            "constructing must not produce"
        );
        assert_eq!(&tensor.bytes().unwrap()[..], &[7, 8]);
        assert_eq!(&tensor.bytes().unwrap()[..], &[7, 8]);
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        // Still usable, and `into_bytes` draws once more.
        assert_eq!(tensor.byte_len(), 2);
        assert_eq!(&tensor.into_bytes().unwrap()[..], &[7, 8]);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    /// A provider failure surfaces verbatim. Naming the tensor is `Writer::materialize`'s
    /// job, not this method's.
    #[test]
    fn bytes_surfaces_a_provider_failure_unannotated() {
        let tensor = Tensor::deferred("w".to_string(), DType::U8, vec![1], None, 1, || {
            Err(Error::IoError("device read failed".to_string()))
        });

        let err = tensor.bytes().unwrap_err();
        assert!(
            matches!(&err, Error::IoError(m) if m == "device read failed"),
            "expected the provider's own error, got {err:?}"
        );
    }

    /// The `Send + Sync` bound on [`ByteSource`] exists so a `Tensor` can reach burn-train's
    /// async checkpointer inside an optimizer record. Nothing in burn-pack's own build would
    /// catch a regression to a non-atomic pointer, so pin it where the type lives.
    #[test]
    #[cfg(target_has_atomic = "ptr")]
    fn tensor_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tensor>();
    }
}
