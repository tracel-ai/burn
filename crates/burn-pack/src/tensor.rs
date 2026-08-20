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

use alloc::format;
use alloc::string::String;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;
// `alloc::sync` needs atomic CAS. A target without it has no threads to share a tensor
// across, so neither the atomic pointer nor the `Send + Sync` bound on the provider is
// needed there, and both are dropped below.
#[cfg(not(target_has_atomic = "ptr"))]
use alloc::rc::Rc as Arc;

use burn_std::{Bytes, DType, Shape};

use crate::base::Error;

/// Shared handle to a deferred tensor's byte provider, kept behind a pointer so [`Tensor`]
/// stays [`Clone`].
///
/// The provider is `Send + Sync` on targets with threads, because a [`Tensor`] must stay
/// `Send`: an optimizer record holds a `Vec` of them and crosses threads through burn-train's
/// async checkpointer, whose `Checkpoint` bound is `Send`. `Sync` follows from sharing the
/// provider at all (`Arc<T>: Send` requires `T: Send + Sync`). A target without atomic CAS
/// has no threads and no `alloc::sync`, so neither bound applies or can be met there, and
/// [`Tensor::deferred`] asks for correspondingly less.
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

impl Source {
    /// Number of bytes described, without producing them.
    fn len(&self) -> usize {
        match self {
            Self::Resident(bytes) => bytes.len(),
            Self::Deferred { len, .. } => *len,
        }
    }

    fn to_bytes(&self) -> Result<Bytes, Error> {
        match self {
            Self::Resident(bytes) => Ok(bytes.clone()),
            Self::Deferred { len, provider } => Self::checked(*len, provider()?),
        }
    }

    fn into_bytes(self) -> Result<Bytes, Error> {
        match self {
            Self::Resident(bytes) => Ok(bytes),
            Self::Deferred { len, provider } => Self::checked(len, provider()?),
        }
    }

    /// Reject a provider that hands back a different length than it declared.
    ///
    /// Every materialization path goes through here, so the length a caller reads from
    /// [`Tensor::byte_len`] is the length they will get. `Writer` commits its offset table
    /// from that number before any provider runs, and a disagreement would misplace every
    /// tensor written after this one.
    fn checked(expected: usize, bytes: Bytes) -> Result<Bytes, Error> {
        if bytes.len() != expected {
            return Err(Error::TensorBytesSizeMismatch(format!(
                "deferred source has inconsistent length (expected {}, got {})",
                expected,
                bytes.len()
            )));
        }
        Ok(bytes)
    }
}

/// A single tensor in a burnpack container, decoupled from any tensor library.
///
/// The bytes are in little-endian layout, and are reached with [`to_bytes`](Self::to_bytes),
/// [`into_bytes`](Self::into_bytes), or [`into_parts`](Self::into_parts). Their length is
/// available up front via
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
    #[cfg(target_has_atomic = "ptr")]
    pub fn deferred(
        name: String,
        dtype: DType,
        shape: impl Into<Shape>,
        param_id: Option<u64>,
        byte_len: usize,
        provider: impl Fn() -> Result<Bytes, Error> + Send + Sync + 'static,
    ) -> Self {
        Self::with_provider(name, dtype, shape, param_id, byte_len, Arc::new(provider))
    }

    /// Create a tensor whose bytes are produced on demand.
    ///
    /// See the `target_has_atomic = "ptr"` variant for the contract. This one drops the
    /// `Send + Sync` bound, which nothing on a single-threaded target can satisfy or needs.
    #[cfg(not(target_has_atomic = "ptr"))]
    pub fn deferred(
        name: String,
        dtype: DType,
        shape: impl Into<Shape>,
        param_id: Option<u64>,
        byte_len: usize,
        provider: impl Fn() -> Result<Bytes, Error> + 'static,
    ) -> Self {
        Self::with_provider(name, dtype, shape, param_id, byte_len, Arc::new(provider))
    }

    /// The half of [`deferred`](Self::deferred) that does not vary by target.
    fn with_provider(
        name: String,
        dtype: DType,
        shape: impl Into<Shape>,
        param_id: Option<u64>,
        byte_len: usize,
        provider: Provider,
    ) -> Self {
        Self {
            name,
            dtype,
            shape: shape.into(),
            param_id,
            source: Source::Deferred {
                len: byte_len,
                provider,
            },
        }
    }

    /// Number of raw bytes the tensor occupies, known without producing them.
    pub fn byte_len(&self) -> usize {
        self.source.len()
    }

    /// The tensor's raw little-endian bytes, leaving it intact.
    ///
    /// Named `to_` rather than `bytes` because it can cost the whole tensor: a `Deferred` one
    /// re-runs its provider (nothing is cached), and a `Resident` one is deep-copied unless
    /// its [`Bytes`] can share their backing. Prefer [`into_bytes`](Self::into_bytes), or
    /// [`into_parts`](Self::into_parts) when the metadata is wanted too.
    pub fn to_bytes(&self) -> Result<Bytes, Error> {
        self.source.to_bytes()
    }

    /// Take the tensor's raw little-endian bytes, producing them if deferred.
    ///
    /// Infallible in practice for a tensor from a [`Reader`](crate::Reader), which only
    /// produces resident tensors; the [`Result`] is there for deferred ones, whose provider
    /// can fail.
    pub fn into_bytes(self) -> Result<Bytes, Error> {
        self.source.into_bytes()
    }

    /// Split into `(name, dtype, shape, param_id, bytes)`, producing the bytes if deferred.
    ///
    /// The fields are public, but taking the bytes consumes the tensor, so reading both
    /// otherwise means cloning the metadata first. This hands over everything in one move.
    pub fn into_parts(self) -> Result<(String, DType, Shape, Option<u64>, Bytes), Error> {
        let Self {
            name,
            dtype,
            shape,
            param_id,
            source,
        } = self;
        Ok((name, dtype, shape, param_id, source.into_bytes()?))
    }
}

// The counting tests below need `fetch_add`, so the whole module wants atomic CAS. Tests are
// only ever run on a host anyway; this keeps `--tests` compiling for embedded targets.
#[cfg(all(test, target_has_atomic = "ptr"))]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    /// `to_bytes` borrows rather than consuming, so the tensor can still be drawn from
    /// afterwards. That is the whole reason it exists next to `into_bytes`.
    #[test]
    fn to_bytes_leaves_a_resident_tensor_intact() {
        let tensor = Tensor::new(
            "w".to_string(),
            DType::U8,
            vec![4],
            None,
            Bytes::from_bytes_vec(vec![1, 2, 3, 4]),
        );

        assert_eq!(&tensor.to_bytes().unwrap()[..], &[1, 2, 3, 4]);
        assert_eq!(&tensor.into_bytes().unwrap()[..], &[1, 2, 3, 4]);
    }

    /// Nothing is cached, so each call runs the provider again. That is why the record load
    /// paths use `into_parts` instead.
    #[test]
    fn to_bytes_reruns_a_deferred_provider_on_every_call() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();

        let tensor = Tensor::deferred("w".to_string(), DType::U8, vec![2], None, 2, move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(Bytes::from_bytes_vec(vec![7, 8]))
        });

        assert_eq!(&tensor.to_bytes().unwrap()[..], &[7, 8]);
        assert_eq!(&tensor.to_bytes().unwrap()[..], &[7, 8]);
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        // Still usable, and `into_bytes` draws once more.
        assert_eq!(tensor.byte_len(), 2);
        assert_eq!(&tensor.into_bytes().unwrap()[..], &[7, 8]);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    /// A provider failure surfaces verbatim. Naming the tensor is `Writer::materialize`'s
    /// job, not this method's.
    #[test]
    fn to_bytes_surfaces_a_provider_failure_unannotated() {
        let tensor = Tensor::deferred("w".to_string(), DType::U8, vec![1], None, 1, || {
            Err(Error::IoError("device read failed".to_string()))
        });

        let err = tensor.to_bytes().unwrap_err();
        assert!(
            matches!(&err, Error::IoError(m) if m == "device read failed"),
            "expected the provider's own error, got {err:?}"
        );
    }

    /// Hands over metadata and bytes in one move, which is what lets the record load paths
    /// avoid cloning the name just to reach the bytes.
    #[test]
    fn into_parts_yields_metadata_and_bytes_together() {
        let tensor = Tensor::deferred(
            "layer.w".to_string(),
            DType::U8,
            vec![2],
            Some(7),
            2,
            || Ok(Bytes::from_bytes_vec(vec![7, 8])),
        );

        let (name, dtype, shape, param_id, bytes) = tensor.into_parts().unwrap();
        assert_eq!(name, "layer.w");
        assert_eq!(dtype, DType::U8);
        assert_eq!(shape.to_vec(), vec![2]);
        assert_eq!(param_id, Some(7));
        assert_eq!(&bytes[..], &[7, 8]);
    }

    /// The `Send + Sync` bound on the provider exists so a `Tensor` can reach burn-train's
    /// async checkpointer inside an optimizer record. Nothing in burn-pack's own build would
    /// catch a regression to a non-atomic pointer, so pin it where the type lives.
    #[test]
    fn tensor_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tensor>();
    }
}
