#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Burn ndarray backend.

#[cfg(any(
    feature = "blas-netlib",
    feature = "blas-openblas",
    feature = "blas-openblas-system",
))]
extern crate blas_src;

mod backend;
mod element;
mod ops;
mod rand;
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
    use alloc::vec::Vec;

    burn_tensor::testgen_all!();

    #[cfg(feature = "std")]
    burn_autodiff::testgen_all!();

    // Quantization
    burn_tensor::testgen_calibration!();
    burn_tensor::testgen_scheme!();
    burn_tensor::testgen_quantize!();
    burn_tensor::testgen_q_data!();
}
