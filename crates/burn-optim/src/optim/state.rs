//! Decomposition of an optimizer state into named tensors and scalars for the burnpack format.
//!
//! Optimizer state is keyed per-[`ParamId`](burn_core::module::ParamId) rather than per module
//! path, and each parameter's state is usually a small struct holding a few tensors plus scalar
//! bookkeeping (e.g. a step counter). [`RecordState`] flattens such a struct into a flat list of
//! named tensors plus a few typed [scalars](burn_pack::Scalar), and reconstructs it on load.
//!
//! Implementations are generated with `#[derive(RecordState)]`; the derive supports the field
//! shapes `Tensor<D>`, `Option<Tensor<D>>`, `Vec<Tensor<D>>`, scalars, optional scalars, nested
//! states and optional nested states. Scalar fields rely on `burn_pack`'s [`From`]/[`TryFrom`]
//! conversions to [`Scalar`](burn_pack::Scalar).

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use burn_core::tensor::{Device, TensorData};
use burn_pack::Scalar;

pub use burn_derive::RecordState;

/// Join a `prefix` and a `leaf` into a dot-separated path (`"prefix.leaf"`).
///
/// An empty prefix yields the leaf unchanged, so a top-level call can pass `""`.
pub fn join_path(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        return String::from(leaf);
    }
    let mut path = String::with_capacity(prefix.len() + 1 + leaf.len());
    path.push_str(prefix);
    path.push('.');
    path.push_str(leaf);
    path
}

/// Join a `prefix` and a numeric `index` into a dot-separated path (`"prefix.3"`).
///
/// Used by the derive to name the elements of a `Vec<Tensor>` field.
pub fn join_index(prefix: &str, index: usize) -> String {
    format!("{prefix}.{index}")
}

/// Accumulates the named tensors and scalars produced while flattening a [`RecordState`].
#[derive(Default, Debug)]
pub struct StateSink {
    /// The collected `(name, data)` tensor leaves.
    pub tensors: Vec<(String, TensorData)>,
    /// The collected `(name, value)` scalar leaves.
    pub scalars: Vec<(String, Scalar)>,
}

impl StateSink {
    /// Record a tensor leaf named `{prefix}.{leaf}`.
    pub fn push_tensor(&mut self, prefix: &str, leaf: &str, data: TensorData) {
        self.tensors.push((join_path(prefix, leaf), data));
    }

    /// Record a scalar leaf named `{prefix}.{leaf}`.
    pub fn push_scalar(&mut self, prefix: &str, leaf: &str, value: Scalar) {
        self.scalars.push((join_path(prefix, leaf), value));
    }
}

/// Provides the named tensors and scalars consumed while reconstructing an [`RecordState`].
///
/// Tensors are taken (removed) by name so the same source can feed several parameters in turn;
/// scalars are looked up by name and left in place.
#[derive(Default, Debug)]
pub struct StateSource {
    tensors: BTreeMap<String, TensorData>,
    scalars: BTreeMap<String, Scalar>,
}

impl StateSource {
    /// Create a source from an existing scalar map (e.g. the burnpack scalars).
    pub fn new(scalars: BTreeMap<String, Scalar>) -> Self {
        Self {
            tensors: BTreeMap::new(),
            scalars,
        }
    }

    /// Register a tensor under its full `name`.
    pub fn insert_tensor(&mut self, name: String, data: TensorData) {
        self.tensors.insert(name, data);
    }

    /// Take the tensor named `{prefix}.{leaf}`, if present.
    pub fn take_tensor(&mut self, prefix: &str, leaf: &str) -> Option<TensorData> {
        self.tensors.remove(&join_path(prefix, leaf))
    }

    /// Read the scalar named `{prefix}.{leaf}`, if present.
    pub fn take_scalar(&mut self, prefix: &str, leaf: &str) -> Option<Scalar> {
        self.scalars.get(&join_path(prefix, leaf)).copied()
    }
}

/// A type that can be flattened into named tensors and scalars and rebuilt from them.
///
/// Generated with `#[derive(RecordState)]`. The `prefix` threads the parameter identity (and any
/// nested field path) through the recursion; leaves are named `{prefix}.{field}`.
pub trait RecordState: Sized {
    /// Flatten `self` into `out`, naming every leaf under `prefix`.
    fn state_flatten(&self, prefix: &str, out: &mut StateSink);

    /// Rebuild a value from `src`, reading leaves named under `prefix`.
    ///
    /// Returns `None` when a required leaf is absent — used so an optional nested state is `None`
    /// exactly when none of its tensors were recorded.
    fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self>;
}

/// The empty state — for stateless values (e.g. a constant learning rate scheduler).
impl RecordState for () {
    fn state_flatten(&self, _prefix: &str, _out: &mut StateSink) {}

    fn state_unflatten(_prefix: &str, _src: &mut StateSource, _device: &Device) -> Option<Self> {
        Some(())
    }
}
