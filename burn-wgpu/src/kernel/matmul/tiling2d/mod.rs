mod base;
mod padding;

/// Loading is done in a continuous manner
pub mod continuous;
/// Loading is done in a continuous manner. lhs is transposed
pub mod continuous_vectorized;
/// Loading is done in a tile manner
pub mod tile;
/// Loading is done in a tile manner. lhs is transposed
pub mod tile_vectorized;

pub use tile_vectorized::*;
