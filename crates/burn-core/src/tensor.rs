pub use burn_tensor::*;

#[cfg(feature = "sparse")]
pub mod sparse {
    pub use burn_sparse::backend::*;
}
