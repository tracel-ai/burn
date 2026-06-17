//! Decomposition of a state struct into named tensors and scalars for the burnpack format.
//!
//! Used for both optimizer state — keyed per-[`ParamId`](burn_core::module::ParamId) rather than
//! per module path — and learning-rate scheduler state. Each state is usually a small struct
//! holding a few tensors plus scalar bookkeeping (e.g. a step counter). [`RecordState`] flattens
//! such a struct into a flat list of named tensors plus a few typed [scalars](burn_pack::Scalar),
//! and reconstructs it on load.
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

    /// Whether any tensor or scalar leaf was recorded under `prefix` (a key beginning with
    /// `"{prefix}."`).
    ///
    /// Used by the derive to tell an absent optional nested state (nothing recorded) apart from a
    /// present one whose leaves all happen to be optional.
    pub fn has_under(&self, prefix: &str) -> bool {
        let pat = join_path(prefix, "");
        self.tensors.keys().any(|k| k.starts_with(&pat))
            || self.scalars.keys().any(|k| k.starts_with(&pat))
    }
}

/// A type that can be flattened into named tensors and scalars and rebuilt from them.
///
/// Generated with `#[derive(RecordState)]`. The `prefix` threads the parameter identity (and any
/// nested field path) through the recursion; leaves are named `{prefix}.{field}`.
pub trait RecordState: Sized + Send + Sync + 'static {
    /// Flatten `self` into `out`, naming every leaf under `prefix`.
    fn state_flatten(&self, prefix: &str, out: &mut StateSink);

    /// Rebuild a value from `src`, reading leaves named under `prefix`.
    ///
    /// Returns `None` when a required leaf is absent. The derive uses [`StateSource::has_under`] to
    /// presence-test optional nested states, so an `Option<Nested>` field is `None` exactly when
    /// nothing was recorded under its path.
    ///
    /// A scalar leaf that is present but holds an incompatible [`Scalar`] variant (e.g. a
    /// hand-edited or forward-version file) is treated the same as absent.
    fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self>;
}

/// The empty state — for stateless values (e.g. a constant learning rate scheduler).
impl RecordState for () {
    fn state_flatten(&self, _prefix: &str, _out: &mut StateSink) {}

    fn state_unflatten(_prefix: &str, _src: &mut StateSource, _device: &Device) -> Option<Self> {
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_core as burn;

    /// Flatten `state` then rebuild it from the produced tensors and scalars.
    fn round_trip<T: RecordState>(state: &T) -> Option<T> {
        let mut sink = StateSink::default();
        state.state_flatten("p", &mut sink);

        let scalars: BTreeMap<String, Scalar> = sink.scalars.into_iter().collect();
        let mut source = StateSource::new(scalars);
        for (name, data) in sink.tensors {
            source.insert_tensor(name, data);
        }

        T::state_unflatten("p", &mut source, &Device::default())
    }

    fn tensor(values: &[f32]) -> Tensor<1> {
        Tensor::from_data(TensorData::from(values), &Device::default())
    }

    fn data(t: &Tensor<1>) -> Vec<f32> {
        t.clone().into_data().to_vec().unwrap()
    }

    #[derive(RecordState, Clone, Debug)]
    struct Inner<const D: usize> {
        weight: Tensor<D>,
        step: i64,
    }

    #[derive(RecordState, Clone, Debug)]
    struct Full<const D: usize> {
        t: Tensor<D>,
        opt_tensor: Option<Tensor<D>>,
        history: Vec<Tensor<D>>,
        count: usize,
        opt_scalar: Option<f64>,
        nested: Inner<D>,
        opt_nested: Option<Inner<D>>,
    }

    #[test]
    fn all_field_kinds_round_trip() {
        let state = Full::<1> {
            t: tensor(&[1.0, 2.0]),
            opt_tensor: Some(tensor(&[3.0])),
            history: vec![tensor(&[4.0]), tensor(&[5.0, 6.0])],
            count: 7,
            opt_scalar: Some(8.5),
            nested: Inner {
                weight: tensor(&[9.0]),
                step: 10,
            },
            opt_nested: Some(Inner {
                weight: tensor(&[11.0]),
                step: 12,
            }),
        };

        let out = round_trip(&state).unwrap();

        assert_eq!(data(&out.t), vec![1.0, 2.0]);
        assert_eq!(data(&out.opt_tensor.unwrap()), vec![3.0]);
        assert_eq!(out.history.len(), 2);
        assert_eq!(data(&out.history[0]), vec![4.0]);
        assert_eq!(data(&out.history[1]), vec![5.0, 6.0]);
        assert_eq!(out.count, 7);
        assert_eq!(out.opt_scalar, Some(8.5));
        assert_eq!(data(&out.nested.weight), vec![9.0]);
        assert_eq!(out.nested.step, 10);
        let opt_nested = out.opt_nested.unwrap();
        assert_eq!(data(&opt_nested.weight), vec![11.0]);
        assert_eq!(opt_nested.step, 12);
    }

    #[test]
    fn absent_optionals_round_trip_to_none() {
        let state = Full::<1> {
            t: tensor(&[1.0]),
            opt_tensor: None,
            history: vec![],
            count: 0,
            opt_scalar: None,
            nested: Inner {
                weight: tensor(&[2.0]),
                step: 0,
            },
            opt_nested: None,
        };

        let out = round_trip(&state).unwrap();

        assert!(out.opt_tensor.is_none());
        assert!(out.history.is_empty());
        assert!(out.opt_scalar.is_none());
        assert!(out.opt_nested.is_none());
    }

    /// Regression: an optional nested state whose fields are all optional must come back `None`
    /// when nothing was recorded under its path — it must not be resurrected as `Some`.
    #[derive(RecordState, Clone, Debug)]
    struct AllOptional<const D: usize> {
        x: Option<Tensor<D>>,
        y: Option<f64>,
    }

    #[derive(RecordState, Clone, Debug)]
    struct OuterOpt<const D: usize> {
        inner: Option<AllOptional<D>>,
    }

    #[test]
    fn optional_all_optional_nested_stays_none() {
        let state = OuterOpt::<1> { inner: None };
        let out = round_trip(&state).unwrap();
        assert!(out.inner.is_none());
    }

    #[test]
    fn optional_all_optional_nested_present_with_content() {
        let state = OuterOpt::<1> {
            inner: Some(AllOptional {
                x: Some(tensor(&[1.0, 2.0])),
                y: None,
            }),
        };
        let out = round_trip(&state).unwrap();
        let inner = out.inner.expect("present because a leaf was recorded");
        assert_eq!(data(&inner.x.unwrap()), vec![1.0, 2.0]);
        assert!(inner.y.is_none());
    }

    #[test]
    fn missing_required_tensor_yields_none() {
        // A source with nothing recorded cannot rebuild a struct that has a required tensor leaf.
        let mut source = StateSource::new(BTreeMap::new());
        assert!(Inner::<1>::state_unflatten("p", &mut source, &Device::default()).is_none());
    }
}
