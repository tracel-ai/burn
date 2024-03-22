#![warn(missing_docs)]
#![allow(unused)] // TODO remove when backend filled

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
    extern crate alloc;
    use super::*;

    pub type TestBackend = Candle<f32, i64>;
    pub type ReferenceBackend = burn_tch::LibTorch<f32>;

    pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
    pub type ReferenceTensor<const D: usize> = burn_tensor::Tensor<ReferenceBackend, D>;
    pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> =
        burn_tensor::Tensor<TestBackend, D, burn_tensor::Bool>;

    type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;
    type TestAutodiffTensor<const D: usize> = burn_tensor::Tensor<TestAutodiffBackend, D>;

    // test activation
    burn_tensor::testgen_gelu!();
    burn_tensor::testgen_prelu!();
    burn_tensor::testgen_relu!();
    burn_tensor::testgen_softmax!();
    burn_tensor::testgen_sigmoid!();
    burn_tensor::testgen_silu!();

    // test module
    burn_tensor::testgen_module_forward!();
    burn_tensor::testgen_module_conv1d!();
    burn_tensor::testgen_module_nearest_interpolate!();
    // burn_tensor::testgen_module_conv2d!();
    // burn_tensor::testgen_module_conv_transpose1d!();
    // burn_tensor::testgen_module_conv_transpose2d!();
    // burn_tensor::testgen_module_max_pool1d!();
    // burn_tensor::testgen_module_max_pool2d!();
    // burn_tensor::testgen_module_avg_pool1d!();
    // burn_tensor::testgen_module_avg_pool2d!();
    // burn_tensor::testgen_module_adaptive_avg_pool1d!();
    // burn_tensor::testgen_module_adaptive_avg_pool2d!();

    // test ops
    burn_tensor::testgen_add!();
    // burn_tensor::testgen_aggregation!();
    burn_tensor::testgen_arange!();
    burn_tensor::testgen_arange_step!();
    burn_tensor::testgen_arg!();
    burn_tensor::testgen_bool!();
    burn_tensor::testgen_cast!();
    burn_tensor::testgen_cat!();
    burn_tensor::testgen_recip!();
    burn_tensor::testgen_clamp!();
    burn_tensor::testgen_cos!();
    burn_tensor::testgen_close!();
    // burn_tensor::testgen_div!();
    burn_tensor::testgen_erf!();
    burn_tensor::testgen_exp!();
    burn_tensor::testgen_flatten!();
    burn_tensor::testgen_full!();
    burn_tensor::testgen_gather_scatter!();
    burn_tensor::testgen_init!();
    burn_tensor::testgen_log!();
    burn_tensor::testgen_log1p!();
    burn_tensor::testgen_map_comparison!();
    burn_tensor::testgen_mask!();
    burn_tensor::testgen_matmul!();
    burn_tensor::testgen_maxmin!();
    burn_tensor::testgen_mul!();
    burn_tensor::testgen_neg!();
    burn_tensor::testgen_permute!();
    burn_tensor::testgen_flip!();
    burn_tensor::testgen_argwhere_nonzero!();
    burn_tensor::testgen_sign!();

    // TODO: https://github.com/tracel-ai/burn/issues/1237
    //
    // burn_tensor::testgen_powf_scalar!();
    // burn_tensor::testgen_powf!();

    burn_tensor::testgen_random!();
    burn_tensor::testgen_repeat!();
    burn_tensor::testgen_reshape!();
    burn_tensor::testgen_select!();
    burn_tensor::testgen_sin!();
    burn_tensor::testgen_slice!();
    burn_tensor::testgen_sqrt!();
    burn_tensor::testgen_abs!();
    burn_tensor::testgen_squeeze!();
    burn_tensor::testgen_sub!();
    burn_tensor::testgen_tanh!();
    burn_tensor::testgen_transpose!();
    burn_tensor::testgen_expand!();

    // test stats
    burn_tensor::testgen_var!();
    burn_tensor::testgen_display!();

    // Behavior
    // burn_autodiff::testgen_ad_broadcast!();

    // Activation
    burn_autodiff::testgen_ad_relu!();
    burn_autodiff::testgen_ad_gelu!();

    // Modules
    // burn_autodiff::testgen_ad_conv1d!();
    // burn_autodiff::testgen_ad_conv2d!();
    // burn_autodiff::testgen_ad_conv_transpose1d!();
    // burn_autodiff::testgen_ad_conv_transpose2d!();
    // burn_autodiff::testgen_ad_max_pool1d!();
    // burn_autodiff::testgen_ad_max_pool2d!();
    // burn_autodiff::testgen_ad_avg_pool1d!();
    // burn_autodiff::testgen_ad_avg_pool2d!();
    // burn_autodiff::testgen_ad_adaptive_avg_pool1d!();
    // burn_autodiff::testgen_ad_adaptive_avg_pool2d!();
    burn_autodiff::testgen_module_backward!();

    // Tensor
    burn_autodiff::testgen_ad_complex!();
    burn_autodiff::testgen_ad_multithread!();
    burn_autodiff::testgen_ad_add!();
    burn_autodiff::testgen_ad_aggregation!();
    burn_autodiff::testgen_ad_maxmin!();
    // burn_autodiff::testgen_ad_cat!();
    burn_autodiff::testgen_ad_cos!();
    burn_autodiff::testgen_ad_cross_entropy_loss!();
    burn_autodiff::testgen_ad_div!();
    burn_autodiff::testgen_ad_erf!();
    burn_autodiff::testgen_ad_exp!();
    burn_autodiff::testgen_ad_slice!();
    burn_autodiff::testgen_ad_gather_scatter!();
    burn_autodiff::testgen_ad_select!();
    burn_autodiff::testgen_ad_log!();
    burn_autodiff::testgen_ad_log1p!();
    burn_autodiff::testgen_ad_mask!();
    burn_autodiff::testgen_ad_matmul!();
    burn_autodiff::testgen_ad_mul!();
    burn_autodiff::testgen_ad_neg!();
    burn_autodiff::testgen_ad_recip!();
    burn_autodiff::testgen_ad_reshape!();
    burn_autodiff::testgen_ad_sin!();
    burn_autodiff::testgen_ad_softmax!();
    burn_autodiff::testgen_ad_sqrt!();
    burn_autodiff::testgen_ad_abs!();
    burn_autodiff::testgen_ad_sub!();
    burn_autodiff::testgen_ad_tanh!();
    burn_autodiff::testgen_ad_transpose!();
    burn_autodiff::testgen_ad_expand!();
}
