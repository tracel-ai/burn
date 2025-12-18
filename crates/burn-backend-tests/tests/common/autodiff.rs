/// Burn autodiff tests, reusable with element types.
pub use super::*;

#[path = "../autodiff/mod.rs"]
mod base;

mod checkpointing {
    pub use super::*;
    use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;

    // Override type def
    pub type TestAutodiffBackend = Autodiff<TestBackend, BalancedCheckpointing>;
    pub type TestAutodiffTensor<const D: usize> = Tensor<TestAutodiffBackend, D>;

    include!("../autodiff/mod.rs");
}

#[cfg(any(
    feature = "vulkan",
    // feature = "cuda", // TODO
    // feature = "rocm",
    feature = "metal"
))]
mod f16 {
    pub type FloatElemType = burn_tensor::f16;
    #[allow(unused)]
    pub use super::IntElemType;

    mod ty {
        include!("backend.rs");
        include!("../autodiff/mod.rs");
    }
}
