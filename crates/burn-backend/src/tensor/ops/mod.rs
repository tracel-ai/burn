mod autodiff;
mod base;
mod bool;
mod float;
mod int;
mod numeric;
mod ordered;

pub use autodiff::*;
pub use base::*;
pub use numeric::*;
pub use ordered::*;

/// Computation to be used to update the existing values in indexed assignment operations (scatter/select).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexingUpdateOp {
    // Assign,
    /// Performs an addition.
    Add,
    // Mul
}

/// Reduction mode for multi-dimensional scatter (scatter_nd).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ScatterNdReduction {
    /// Overwrite existing values.
    Assign,
    /// Add values to existing.
    Add,
    /// Multiply existing values.
    Mul,
    /// Take element-wise minimum.
    Min,
    /// Take element-wise maximum.
    Max,
}
