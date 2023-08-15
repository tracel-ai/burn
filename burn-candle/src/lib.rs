#![warn(missing_docs)]

//! Burn Candle Backend

#[macro_use]
extern crate derive_new;

mod backend;
mod element;
mod ops;
mod tensor;

pub use backend::*;
pub use tensor::*;

#[cfg(test)]
mod tests {
    use super::*;

    pub type TestBackend = CandleBackend<f32>;
    pub type ReferenceBackend = burn_tch::TchBackend<f32>;

    pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    pub type ReferenceTensor<const D: usize> = burn_tensor::Tensor<ReferenceBackend, D>;
    pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;

    burn_tensor::testgen_all!();
    burn_autodiff::testgen_all!();
}
