mod autodiff;
mod base;
mod bool;
mod float;
mod int;
mod numeric;
mod orderable;

pub use autodiff::*;
pub use base::*;
pub use numeric::*;
pub use orderable::*;

/// Computation to be used to update the existing values in indexed assignment operations (scatter/select).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexingUpdateOp {
    // Assign,
    /// Performs an addition.
    Add,
    // Mul
}
