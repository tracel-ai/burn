#![warn(missing_docs)]
#![allow(clippy::single_range_in_vec_init)]

//! Burn Tch Backend

mod backend;
mod element;
mod ops;
mod tensor;

pub use backend::*;
pub use element::*;
pub use tensor::*;

#[cfg(test)]
mod tests {
    type TestBackend = crate::TchBackend<f32>;
    type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;

    burn_tensor::testgen_all!();
    burn_autodiff::testgen_all!();
}
