mod base;
mod padding;

/// Loading to shared memory is done in a contiguous manner
pub mod contiguous;
/// Loading is done in a contiguous manner, with left hand tensor being transposed.
pub mod contiguous_vectorized;
/// Loading is done in a tile manner
pub mod tile;
/// Loading is done in a tile manner, with left hand tensor being transposed.
pub mod tile_vectorized;

pub use contiguous_vectorized::*;
