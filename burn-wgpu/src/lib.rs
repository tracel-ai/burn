#![warn(missing_docs)]

//! Burn WGPU Backend

#[macro_use]
extern crate derive_new;
extern crate alloc;

mod ops;

/// Compute related module.
pub mod compute;
/// Kernel module
pub mod kernel;
/// Tensor module.
pub mod tensor;

mod element;
pub use element::{FloatElement, IntElement};

mod device;
pub use device::*;

mod backend;
pub use backend::*;

mod graphics;
pub use graphics::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    type GraphicsApi = Metal;

    #[cfg(not(target_os = "macos"))]
    type GraphicsApi = Vulkan;

    pub type TestBackend = WgpuBackend<GraphicsApi, f32, i32>;
    pub type ReferenceBackend = burn_ndarray::NdArrayBackend<f32>;

    pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    pub type ReferenceTensor<const D: usize> = burn_tensor::Tensor<ReferenceBackend, D>;
    pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;

    burn_tensor::testgen_all!();
    burn_autodiff::testgen_all!();
}
