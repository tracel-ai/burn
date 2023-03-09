#[cfg(any(
    feature = "blas-accelerate",
    features = "blas-netlib",
    feature = "blas-openblas",
    feature = "blas-openblas-system"
))]
mod blas;

mod base;
mod matrixmultiply;

pub use base::*;
