mod binary;
mod binary_int;
mod cast;
mod clamp;
mod comparison;
mod contiguous;
mod cross;
mod index;
mod mask;
mod unary_float;
mod unary_int;
mod unary_numeric;

pub(crate) use binary::*;
pub(crate) use binary_int::*;
pub use cast::*;
pub use contiguous::*;
pub(crate) use cross::*;
pub use mask::*;
pub(crate) use unary_float::*;
pub(crate) use unary_int::*;
pub(crate) use unary_numeric::*;

pub use crate::cubecl::prelude::KernelMetadata;

/// Attention kernels
pub mod attention;
/// Convolution kernels
pub mod conv;
/// Interpolation kernels
pub mod interpolate;
/// Matmul kernels
pub mod matmul;
/// Pooling kernels
pub mod pool;
/// Pseudo-random number generator kernels
pub mod prng;
/// Quantization operations
pub mod quantization;
/// Reduction algorithms
pub mod reduce;

pub(crate) use clamp::*;
pub(crate) use comparison::*;
pub use index::*;

pub(crate) mod utils;
