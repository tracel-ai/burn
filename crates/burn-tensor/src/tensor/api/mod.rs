pub(crate) mod check;

mod argwhere;
mod autodiff;
mod base;
mod bool;
mod cartesian_grid;
mod float;
mod fmod;
mod int;
mod kind;
mod numeric;
mod sort;
mod take;
mod transaction;
mod trunc;

pub use argwhere::argwhere_data;
pub use autodiff::*;
pub use base::*;
pub use cartesian_grid::cartesian_grid;
pub use float::{DEFAULT_ATOL, DEFAULT_RTOL};
pub use kind::*;
pub use numeric::*;
pub use sort::{argsort, sort, sort_with_indices};
pub use transaction::*;

/// Computation to be used to update the existing values in indexed assignment operations (scatter/select).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IndexingUpdateOp {
    // Assign,
    /// Performs an addition.
    Add,
    // Mul
}
