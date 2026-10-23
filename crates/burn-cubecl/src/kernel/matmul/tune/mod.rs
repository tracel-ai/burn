#[cfg(all(feature = "autotune", not(feature = "tile-kernels-only")))]
mod base;

#[cfg(all(feature = "autotune", not(feature = "tile-kernels-only")))]
mod bounds;

#[cfg(feature = "tile-kernels-only")]
mod tile_kernels;

#[cfg(all(feature = "autotune", not(feature = "tile-kernels-only")))]
pub use base::matmul_autotune;
#[cfg(feature = "tile-kernels-only")]
pub use tile_kernels::matmul_autotune;
