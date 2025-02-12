#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Burn ndarray backend.

#[macro_use]
extern crate derive_new;

#[cfg(any(
    feature = "blas-netlib",
    feature = "blas-openblas",
    feature = "blas-openblas-system",
))]
extern crate blas_src;

mod backend;
mod element;
mod ops;
mod sharing;
mod tensor;

pub use backend::*;
pub use element::*;
pub(crate) use sharing::*;
pub use tensor::*;

extern crate alloc;

#[cfg(test)]
mod tests {
    type TestBackend = crate::NdArray<f32>;
    type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;
    type TestTensorBool<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Bool>;

    use alloc::format;
    use alloc::vec;

    burn_tensor::testgen_all!();
    burn_tensor::testgen_quantization!();

    #[cfg(feature = "std")]
    burn_autodiff::testgen_all!();
}
