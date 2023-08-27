#![warn(missing_docs)]

//! Burn WGPU Backend

#[macro_use]
extern crate derive_new;

mod ops;

/// Benchmark module
pub mod benchmark;
/// Kernel module
pub mod kernel;

pub(crate) mod context;
pub(crate) mod pool;
pub(crate) mod tensor;
pub(crate) mod tune;

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
