#![allow(missing_docs)]

mod abs;
mod adaptive_avgpool1d;
mod adaptive_avgpool2d;
mod add;
mod aggregation;
mod avgpool1d;
mod avgpool2d;
mod backward;
mod bridge;
mod broadcast;
mod cat;
mod ceil;
mod checkpoint;
mod complex;
mod conv1d;
mod conv2d;
mod conv3d;
mod conv_transpose1d;
mod conv_transpose2d;
mod conv_transpose3d;
mod cos;
mod cross;
mod cross_entropy;
mod deform_conv2d;
mod div;
mod erf;
mod exp;
mod expand;
mod flip;
mod floor;
mod gather_scatter;
mod gelu;
mod gradients;
mod log;
mod log1p;
mod log_sigmoid;
mod mask;
mod matmul;
mod maxmin;
mod maxpool1d;
mod maxpool2d;
mod memory_management;
mod mul;
mod multithread;
mod nearest_interpolate;
mod neg;
mod nonzero;
mod permute;
mod pow;
mod recip;
mod relu;
mod remainder;
mod repeat_dim;
mod reshape;
mod round;
mod select;
mod sigmoid;
mod sign;
mod sin;
mod slice;
mod slice_assign;
mod softmax;
mod sort;
mod sqrt;
mod sub;
mod tanh;
mod transpose;

#[macro_export]
macro_rules! testgen_all {
    // Avoid using paste dependency with no parameters
    () => {
        mod autodiff {
            pub use super::*;
            type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;
            type TestAutodiffTensor<const D: usize> = burn_tensor::Tensor<TestAutodiffBackend, D>;

            pub type FloatType = <TestBackend as burn_tensor::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as burn_tensor::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as burn_tensor::backend::Backend>::BoolTensorPrimitive;

            $crate::testgen_with_float_param!();
        }
        mod autodiff_checkpointing {
            pub use super::*;
            type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend, burn_autodiff::checkpoint::strategy::BalancedCheckpointing>;
            type TestAutodiffTensor<const D: usize> = burn_tensor::Tensor<TestAutodiffBackend, D>;

            pub type FloatType = <TestBackend as burn_tensor::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as burn_tensor::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as burn_tensor::backend::Backend>::BoolTensorPrimitive;

            $crate::testgen_with_float_param!();
        }
    };
    ([$($float:ident),*]) => {
        mod autodiff_checkpointing {
            pub use super::*;
            type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend, burn_autodiff::checkpoint::strategy::BalancedCheckpointing>;
            type TestAutodiffTensor<const D: usize> = burn_tensor::Tensor<TestAutodiffBackend, D>;

            pub type FloatType = <TestBackend as burn_tensor::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as burn_tensor::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as burn_tensor::backend::Backend>::BoolElem;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<$float, IntType, BoolType>;
                    pub type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend, burn_autodiff::checkpoint::strategy::BalancedCheckpointing>;
                    pub type TestAutodiffTensor<const D: usize> = burn_tensor::Tensor<TestAutodiffBackend, D>;
                    pub type TestTensor<const D: usize> = TestTensor2<$float, IntType, BoolType, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<$float, IntType, BoolType, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<$float, IntType, BoolType, D>;

                    type FloatType = $float;

                    $crate::testgen_with_float_param!();
                })*
            }
        }

        mod autodiff {
            pub use super::*;
            type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;
            type TestAutodiffTensor<const D: usize> = burn_tensor::Tensor<TestAutodiffBackend, D>;

            pub type FloatType = <TestBackend as burn_tensor::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as burn_tensor::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as burn_tensor::backend::Backend>::BoolElem;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<$float, IntType, BoolType>;
                    pub type TestAutodiffBackend = burn_autodiff::Autodiff<TestBackend>;
                    pub type TestAutodiffTensor<const D: usize> = burn_tensor::Tensor<TestAutodiffBackend, D>;
                    pub type TestTensor<const D: usize> = TestTensor2<$float, IntType, BoolType, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<$float, IntType, BoolType, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<$float, IntType, BoolType, D>;

                    type FloatType = $float;

                    $crate::testgen_with_float_param!();
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_with_float_param {
    () => {
        // Behaviour
        burn_autodiff::testgen_ad_broadcast!();
        burn_autodiff::testgen_gradients!();
        burn_autodiff::testgen_bridge!();
        burn_autodiff::testgen_checkpoint!();
        burn_autodiff::testgen_memory_management!();

        // Activation
        burn_autodiff::testgen_ad_relu!();
        burn_autodiff::testgen_ad_gelu!();

        // Modules
        burn_autodiff::testgen_ad_conv1d!();
        burn_autodiff::testgen_ad_conv2d!();
        burn_autodiff::testgen_ad_conv3d!();
        // #[cfg(not(target_os = "macos"))] // Wgpu on MacOS currently doesn't support atomic compare exchange
        // burn_autodiff::testgen_ad_deform_conv2d!(); // This kernel in cubecl isn't implemented without atomics
        burn_autodiff::testgen_ad_conv_transpose1d!();
        burn_autodiff::testgen_ad_conv_transpose2d!();
        burn_autodiff::testgen_ad_conv_transpose3d!();
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
        burn_autodiff::testgen_ad_cross!();
        burn_autodiff::testgen_ad_cross_entropy_loss!();
        burn_autodiff::testgen_ad_div!();
        burn_autodiff::testgen_ad_remainder!();
        burn_autodiff::testgen_ad_erf!();
        burn_autodiff::testgen_ad_exp!();
        burn_autodiff::testgen_ad_slice!();
        burn_autodiff::testgen_ad_slice_assign!();
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
        burn_autodiff::testgen_ad_round!();
        burn_autodiff::testgen_ad_floor!();
        burn_autodiff::testgen_ad_ceil!();
        burn_autodiff::testgen_ad_sigmoid!();
        burn_autodiff::testgen_ad_log_sigmoid!();
        burn_autodiff::testgen_ad_transpose!();
        burn_autodiff::testgen_ad_permute!();
        burn_autodiff::testgen_ad_flip!();
        burn_autodiff::testgen_ad_nonzero!();
        burn_autodiff::testgen_ad_sign!();
        burn_autodiff::testgen_ad_expand!();
        burn_autodiff::testgen_ad_sort!();
        burn_autodiff::testgen_ad_repeat_dim!();
    };
}
