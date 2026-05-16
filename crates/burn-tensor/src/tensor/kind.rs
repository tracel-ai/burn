use burn_dispatch::Dispatch;

use crate::bridge as backend;
pub use crate::bridge::{Bool, Float, Int, TensorKind};

/// The base trait for any tensor kind.
pub trait Basic: backend::BasicOps<Dispatch> {}
impl<K: backend::BasicOps<Dispatch>> Basic for K {}

/// Kinds that support numeric operations.
pub trait Numeric: Basic + backend::Numeric<Dispatch> {}
impl<K: Basic + backend::Numeric<Dispatch>> Numeric for K {}

/// Kinds that support ordered operations.
pub trait Ordered: Numeric + backend::Ordered<Dispatch> {}
impl<K: Numeric + backend::Ordered<Dispatch>> Ordered for K {}

/// Kinds that support float math operations.
pub trait FloatMath: Numeric + backend::FloatMathOps<Dispatch> {}
impl<K: Numeric + backend::FloatMathOps<Dispatch>> FloatMath for K {}

/// Kinds that support transaction operations.
pub trait Transaction: Basic + backend::TransactionOp<Dispatch> {}
impl<K: Basic + backend::TransactionOp<Dispatch>> Transaction for K {}

/// Kinds that support autodiff operations.
// #[cfg(feature = "autodiff")]
pub trait Autodiff: Basic + backend::BasicAutodiffOps<Dispatch> {}
// #[cfg(feature = "autodiff")]
impl<K: backend::BasicAutodiffOps<Dispatch, InnerKind = K>> Autodiff for K {}
