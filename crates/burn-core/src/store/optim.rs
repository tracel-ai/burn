//! Minimal decomposition of an optimizer state into named tensors and scalars.
//!
//! Optimizer state is keyed per-[`ParamId`](crate::module::ParamId) rather than per module path,
//! and each parameter's state is usually a small struct holding a few tensors plus scalar
//! bookkeeping (e.g. a step counter). [`OptimState`] flattens such a struct into a flat list of
//! named tensors (suitable for the [burnpack](burn_pack) format) plus a few typed scalars, and
//! reconstructs it on load.
//!
//! Implementations are generated with `#[derive(OptimState)]`; see the derive for the supported
//! field shapes (`Tensor<D>`, `Option<Tensor<D>>`, `Vec<Tensor<D>>`, scalars, optional scalars,
//! nested states and optional nested states).

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::tensor::{Device, TensorData};

pub use burn_derive::OptimState;
pub use burn_pack::Scalar;

/// A value that can be stored as a burnpack [`Scalar`].
///
/// Implemented for the primitive numeric and boolean types so that `#[derive(OptimState)]` can
/// serialize scalar state fields (step counters, learning rates, flags) without routing them
/// through strings.
pub trait ScalarValue: Sized {
    /// Encode the value as a [`Scalar`].
    fn to_scalar(self) -> Scalar;
    /// Decode the value from a [`Scalar`], returning `None` on a type/range mismatch.
    fn from_scalar(scalar: Scalar) -> Option<Self>;
}

macro_rules! impl_scalar_int {
    ($($t:ty => $variant:ident),* $(,)?) => {
        $(
            impl ScalarValue for $t {
                fn to_scalar(self) -> Scalar {
                    Scalar::$variant(self as _)
                }
                fn from_scalar(scalar: Scalar) -> Option<Self> {
                    match scalar {
                        Scalar::Int(v) => v.try_into().ok(),
                        Scalar::UInt(v) => v.try_into().ok(),
                        _ => None,
                    }
                }
            }
        )*
    };
}

impl_scalar_int!(i8 => Int, i16 => Int, i32 => Int, i64 => Int, isize => Int);
impl_scalar_int!(u8 => UInt, u16 => UInt, u32 => UInt, u64 => UInt, usize => UInt);

impl ScalarValue for f64 {
    fn to_scalar(self) -> Scalar {
        Scalar::Float(self)
    }
    fn from_scalar(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Float(v) => Some(v),
            Scalar::Int(v) => Some(v as f64),
            Scalar::UInt(v) => Some(v as f64),
            _ => None,
        }
    }
}

impl ScalarValue for f32 {
    fn to_scalar(self) -> Scalar {
        Scalar::Float(self as f64)
    }
    fn from_scalar(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Float(v) => Some(v as f32),
            _ => None,
        }
    }
}

impl ScalarValue for bool {
    fn to_scalar(self) -> Scalar {
        Scalar::Bool(self)
    }
    fn from_scalar(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Bool(v) => Some(v),
            _ => None,
        }
    }
}

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

/// Accumulates the named tensors and scalars produced while flattening an [`OptimState`].
#[derive(Default, Debug)]
pub struct OptimStateSink {
    /// The collected `(name, data)` tensor leaves.
    pub tensors: Vec<(String, TensorData)>,
    /// The collected `(name, value)` scalar leaves.
    pub scalars: Vec<(String, Scalar)>,
}

impl OptimStateSink {
    /// Record a tensor leaf named `{prefix}.{leaf}`.
    pub fn push_tensor(&mut self, prefix: &str, leaf: &str, data: TensorData) {
        self.tensors.push((join_path(prefix, leaf), data));
    }

    /// Record a scalar leaf named `{prefix}.{leaf}`.
    pub fn push_scalar(&mut self, prefix: &str, leaf: &str, value: Scalar) {
        self.scalars.push((join_path(prefix, leaf), value));
    }
}

/// Provides the named tensors and scalars consumed while reconstructing an [`OptimState`].
///
/// Tensors are taken (removed) by name so the same source can feed several parameters in turn;
/// scalars are looked up by name and left in place.
#[derive(Default, Debug)]
pub struct OptimStateSource {
    tensors: BTreeMap<String, TensorData>,
    scalars: BTreeMap<String, Scalar>,
}

impl OptimStateSource {
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
/// Generated with `#[derive(OptimState)]`. The `prefix` threads the parameter identity (and any
/// nested field path) through the recursion; leaves are named `{prefix}.{field}`.
pub trait OptimState: Sized {
    /// Flatten `self` into `out`, naming every leaf under `prefix`.
    fn state_flatten(&self, prefix: &str, out: &mut OptimStateSink);

    /// Rebuild a value from `src`, reading leaves named under `prefix`.
    ///
    /// Returns `None` when a required leaf is absent — used so an optional nested state is `None`
    /// exactly when none of its tensors were recorded.
    fn state_unflatten(prefix: &str, src: &mut OptimStateSource, device: &Device) -> Option<Self>;
}
