mod meshgrid;

pub use meshgrid::*;

/// Enum to specify grid sparsity mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridSparsity {
    /// The grid is sparse, expanded only at the cardinal dimensions.
    Sparse,

    /// The grid is fully expanded to the full cartesian product shape.
    Dense,
}

/// Enum to specify index cardinal layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridIndexing {
    /// Dimensions are in the same order as the cardinality of the inputs.
    /// Equivalent to "ij" indexing in NumPy and PyTorch.
    Matrix,

    /// The first two dimensions are swapped.
    /// Equivalent to "xy" indexing in NumPy and PyTorch.
    Cartesian,
}
