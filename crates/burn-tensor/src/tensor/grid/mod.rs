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

/// Legacy compatibility enum for grid indexing modes.
///
/// NumPy (and by copying, PyTorch) used an indexing mode which swapped the first two dimensions,
/// added to simplify some graphics and plotting tasks. As it broke the natural order of the
/// dimensions, the behavior was flagged, and migration plans are in-flight in both libraries
/// to make the natural ordering the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridCompatIndexing {
    /// Dimensions are in the same order as the cardinality of the inputs.
    /// Equivalent to "ij" indexing in NumPy and PyTorch.
    MatrixIndexing,

    /// The first two dimensions are swapped.
    /// Equivalent to "xy" indexing in NumPy and PyTorch.
    CartesianIndexing,
}
