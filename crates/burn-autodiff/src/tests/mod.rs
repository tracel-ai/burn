#![allow(missing_docs)]

mod abs;
mod adaptive_avgpool1d;
mod adaptive_avgpool2d;
mod add;
mod aggregation;
mod avgpool1d;
mod avgpool2d;
mod backward;
mod broadcast;
mod cat;
mod checkpoint;
mod complex;
mod conv1d;
mod conv2d;
mod conv_transpose1d;
mod conv_transpose2d;
mod cos;
mod cross_entropy;
mod div;
mod erf;
mod exp;
mod expand;
mod flip;
mod gather_scatter;
mod gelu;
mod gradients;
mod log;
mod log1p;
mod mask;
mod matmul;
mod maxmin;
mod maxpool1d;
mod maxpool2d;
mod mul;
mod multithread;
mod nearest_interpolate;
mod neg;
mod nonzero;
mod permute;
mod pow;
mod recip;
mod relu;
mod reshape;
mod select;
mod sigmoid;
mod sign;
mod sin;
mod slice;
mod softmax;
mod sort;
mod sqrt;
mod sub;
mod tanh;
mod transpose;

#[macro_export]
macro_rules! testgen_all {
    () => {
        type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;
        type TestAutodiffTensor<const D: usize> = burn_tensor::Tensor<TestAutodiffBackend, D>;

        // Behavior
        burn_autodiff::testgen_ad_broadcast!();
        burn_autodiff::testgen_gradients!();
        burn_autodiff::testgen_checkpoint!();

        // Activation
        burn_autodiff::testgen_ad_relu!();
        burn_autodiff::testgen_ad_gelu!();

        // Modules
        burn_autodiff::testgen_ad_conv1d!();
        burn_autodiff::testgen_ad_conv2d!();
        burn_autodiff::testgen_ad_conv_transpose1d!();
        burn_autodiff::testgen_ad_conv_transpose2d!();
        burn_autodiff::testgen_ad_max_pool1d!();
        burn_autodiff::testgen_ad_max_pool2d!();
        burn_autodiff::testgen_ad_avg_pool1d!();
        burn_autodiff::testgen_ad_avg_pool2d!();
        burn_autodiff::testgen_ad_adaptive_avg_pool1d!();
        burn_autodiff::testgen_ad_adaptive_avg_pool2d!();
        burn_autodiff::testgen_module_backward!();
        burn_autodiff::testgen_ad_nearest_interpolate!();

        // Tensor
        burn_autodiff::testgen_ad_complex!();
        burn_autodiff::testgen_ad_multithread!();
        burn_autodiff::testgen_ad_add!();
        burn_autodiff::testgen_ad_aggregation!();
        burn_autodiff::testgen_ad_maxmin!();
        burn_autodiff::testgen_ad_cat!();
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
        burn_autodiff::testgen_ad_powf!();
        burn_autodiff::testgen_ad_recip!();
        burn_autodiff::testgen_ad_reshape!();
        burn_autodiff::testgen_ad_sin!();
        burn_autodiff::testgen_ad_softmax!();
        burn_autodiff::testgen_ad_sqrt!();
        burn_autodiff::testgen_ad_abs!();
        burn_autodiff::testgen_ad_sub!();
        burn_autodiff::testgen_ad_tanh!();
        burn_autodiff::testgen_ad_sigmoid!();
        burn_autodiff::testgen_ad_transpose!();
        burn_autodiff::testgen_ad_permute!();
        burn_autodiff::testgen_ad_flip!();
        burn_autodiff::testgen_ad_nonzero!();
        burn_autodiff::testgen_ad_sign!();
        burn_autodiff::testgen_ad_expand!();
        burn_autodiff::testgen_ad_sort!();
    };
}
