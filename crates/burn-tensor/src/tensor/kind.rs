use burn_backend::tensor as backend;
use burn_dispatch::Dispatch;

pub use burn_backend::tensor::{Bool, Float, Int, TensorKind};

/// The base trait for any tensor kind.
pub trait Basic: backend::BasicOps<Dispatch> {}
impl<K: backend::BasicOps<Dispatch>> Basic for K {}

/// Kinds that support numeric operations.
pub trait Numeric: backend::Numeric<Dispatch> {}

impl<K: backend::Numeric<Dispatch>> Numeric for K {}

/// Kinds that support ordered operations.
pub trait Ordered: backend::Ordered<Dispatch> + Numeric {}
impl<K: backend::Ordered<Dispatch>> Ordered for K {}

// Not required, should be part of basic ops
// /// Kinds that support autodiff operations.
// pub trait Autodiff: backend::BasicAutodiffOps<Dispatch> + {}
// impl<K: backend::BasicAutodiffOps<Dispatch>> Autodiff for K {}
