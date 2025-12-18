#![allow(
    clippy::single_range_in_vec_init,
    clippy::duplicate_mod,
    reason = "false positive"
)]

/// Burn autodiff tests, reusable with element types.
pub use super::*;

mod backend;
pub use backend::*;

#[path = "../autodiff/mod.rs"]
mod autodiff;

mod autodiff_checkpointing {
    pub use super::*;
    use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;

    // Override type def
    pub type TestAutodiffBackend = Autodiff<TestBackend, BalancedCheckpointing>;
    pub type TestAutodiffTensor<const D: usize> = Tensor<TestAutodiffBackend, D>;

    include!("../autodiff/mod.rs");
}
