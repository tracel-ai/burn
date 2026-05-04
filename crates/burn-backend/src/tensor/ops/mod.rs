mod autodiff;
mod base;
mod bool;
mod float;
mod int;
mod numeric;
mod ordered;

pub use autodiff::*;
pub use base::*;
pub use float::FloatMathOps;
pub use numeric::*;
pub use ordered::*;

/// Computation to be used to update the existing values in indexed assignment operations (scatter/select).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexingUpdateOp {
    /// Overwrite existing values.
    Assign,
    /// Performs an addition.
    Add,
    /// Multiply existing values.
    Mul,
    /// Take element-wise minimum.
    Min,
    /// Take element-wise maximum.
    Max,
}
