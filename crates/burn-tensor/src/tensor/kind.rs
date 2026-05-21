// Sealed traits, limited to Bool, Float and Int types which implement the pub(crate) traits.
#![allow(private_bounds)]

#[cfg(feature = "extension")]
pub use crate::bridge::BridgeTensor;

pub use crate::bridge::{Bool, Float, Int, TensorKind, TensorKindId};

/// The base trait for any tensor kind.
pub trait Basic: crate::ops::BasicOps {}
impl<K: crate::ops::BasicOps> Basic for K {}

/// Kinds that support numeric operations.
pub trait Numeric: Basic + crate::ops::Numeric {}
impl<K: Basic + crate::ops::Numeric> Numeric for K {}

/// Kinds that support ordered operations.
pub trait Ordered: Numeric + crate::ops::Ordered {}
impl<K: Numeric + crate::ops::Ordered> Ordered for K {}

/// Kinds that support float math operations.
pub trait FloatMath: Numeric + crate::ops::FloatMathOps {}
impl<K: Numeric + crate::ops::FloatMathOps> FloatMath for K {}

/// Kinds that support transaction operations.
pub trait Transaction: Basic + crate::ops::TransactionOp {}
impl<K: Basic + crate::ops::TransactionOp> Transaction for K {}

/// Kinds that support autodiff operations.
// #[cfg(feature = "autodiff")]
pub trait Autodiff: Basic + crate::ops::BasicAutodiffOps {}
// #[cfg(feature = "autodiff")]
impl<K: crate::ops::BasicAutodiffOps> Autodiff for K {}
