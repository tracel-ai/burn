mod affine_grid;
mod meshgrid;

pub use meshgrid::*;

pub use affine_grid::*;

/// Enum to specify index cardinal layout.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridIndexing {
    /// Dimensions are in the same order as the cardinality of the inputs.
    /// Equivalent to "ij" indexing in NumPy and PyTorch.
    #[default]
    Matrix,

    /// The same as Matrix, but the first two dimensions are swapped.
    /// Equivalent to "xy" indexing in NumPy and PyTorch.
    Cartesian,
}

/// Enum to specify grid sparsity mode.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridSparsity {
    /// The grid is fully expanded to the full cartesian product shape.
    #[default]
    Dense,

    /// The grid is sparse, expanded only at the cardinal dimensions.
    Sparse,
}

/// Grid policy options.
#[derive(new, Default, Debug, Copy, Clone)]
pub struct GridOptions {
    /// Indexing mode.
    pub indexing: GridIndexing,

    /// Sparsity mode.
    pub sparsity: GridSparsity,
}

impl From<GridIndexing> for GridOptions {
    fn from(value: GridIndexing) -> Self {
        Self {
            indexing: value,
            ..Default::default()
        }
    }
}
impl From<GridSparsity> for GridOptions {
    fn from(value: GridSparsity) -> Self {
        Self {
            sparsity: value,
            ..Default::default()
        }
    }
}

/// Enum to specify the index dimension position.
#[derive(Default, Debug, Copy, Clone)]
pub enum IndexPos {
    /// The index is in the first dimension.
    #[default]
    First,

    /// The index is in the last dimension.
    Last,
}
