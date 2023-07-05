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
pub(crate) mod element;
pub(crate) mod pool;
pub(crate) mod tensor;

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
    pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;

    burn_tensor::testgen_add!();
    burn_tensor::testgen_sub!();
    burn_tensor::testgen_div!();
    burn_tensor::testgen_mul!();
    burn_tensor::testgen_neg!();
    burn_tensor::testgen_powf!();
    burn_tensor::testgen_exp!();
    burn_tensor::testgen_log!();
    burn_tensor::testgen_log1p!();
    burn_tensor::testgen_sqrt!();
    burn_tensor::testgen_cos!();
    burn_tensor::testgen_sin!();
    burn_tensor::testgen_tanh!();
    burn_tensor::testgen_erf!();
    burn_tensor::testgen_relu!();
    burn_tensor::testgen_matmul!();
    burn_tensor::testgen_reshape!();
    burn_tensor::testgen_transpose!();
    burn_tensor::testgen_slice!();
    burn_tensor::testgen_aggregation!();
    burn_tensor::testgen_arg!();
    burn_tensor::testgen_map_comparison!();
    burn_tensor::testgen_arange!();
    burn_tensor::testgen_mask!();
    burn_tensor::testgen_cat!();
    burn_tensor::testgen_select!();
    burn_tensor::testgen_gather_scatter!();

    type TestADBackend = burn_autodiff::ADBackendDecorator<TestBackend>;
    type TestADTensor<const D: usize, K> = burn_tensor::Tensor<TestADBackend, D, K>;

    burn_autodiff::testgen_ad_add!();
    burn_autodiff::testgen_ad_sub!();
    burn_autodiff::testgen_ad_div!();
    burn_autodiff::testgen_ad_mul!();
    burn_autodiff::testgen_ad_neg!();
    burn_autodiff::testgen_ad_powf!();
    burn_autodiff::testgen_ad_exp!();
    burn_autodiff::testgen_ad_log!();
    burn_autodiff::testgen_ad_log1p!();
    burn_autodiff::testgen_ad_sqrt!();
    burn_autodiff::testgen_ad_cos!();
    burn_autodiff::testgen_ad_sin!();
    burn_autodiff::testgen_ad_tanh!();
    burn_autodiff::testgen_ad_matmul!();
    burn_autodiff::testgen_ad_reshape!();
    burn_autodiff::testgen_ad_transpose!();
    burn_autodiff::testgen_ad_slice!();
    burn_autodiff::testgen_ad_aggregation!();
    burn_autodiff::testgen_ad_cat!();
    burn_autodiff::testgen_ad_mask!();
    burn_autodiff::testgen_ad_select!();

    // Once all operations will be implemented.
    // burn_tensor::testgen_all!();
    // burn_autodiff::testgen_all!();
}
