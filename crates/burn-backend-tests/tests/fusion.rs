//! Burn tensor and autodiff tests for CubeCL backends with fusion enabled.

#![allow(
    clippy::single_range_in_vec_init,
    clippy::duplicate_mod,
    reason = "false positive"
)]
extern crate alloc;

#[cfg(feature = "cube")]
#[path = "."]
mod fusion {
    pub type FloatElemType = f32;
    pub type IntElemType = i32;

    #[path = "common/backend.rs"]
    mod backend;
    pub use backend::types::*;

    pub type TestBackend = burn_fusion::Fusion<backend::TestBackend>;
    pub type TestTensor<const D: usize> = Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> = Tensor<TestBackend, D, burn_tensor::Bool>;

    // Tensor tests
    mod tensor {
        include!("common/tensor.rs");
    }

    // Autodiff tests
    mod autodiff {
        include!("common/autodiff.rs");
    }
}
