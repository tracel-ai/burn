mod activation;
mod clone_invariance;
mod module;
mod ops;
mod primitive;
mod quantization;
mod stats;

pub use cubecl::prelude::{Float, Int, Numeric};

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_all {
    () => {
        pub mod tensor {
            pub use super::*;

            pub type FloatType = <TestBackend as $crate::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as $crate::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as $crate::backend::Backend>::BoolElem;

            $crate::testgen_with_float_param!();
            $crate::testgen_no_param!();
        }
    };
    ([$($float:ident),*], [$($int:ident),*], [$($bool:ident),*]) => {
        pub mod tensor {
            pub use super::*;

            pub type FloatType = <TestBackend as $crate::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as $crate::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as $crate::backend::Backend>::BoolElem;

            ::paste::paste! {
                $(mod [<$float _ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<$float, IntType, BoolType>;
                    pub type TestTensor<const D: usize> = TestTensor2<$float, IntType, BoolType, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<$float, IntType, BoolType, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<$float, IntType, BoolType, D>;

                    pub type FloatType = $float;

                    $crate::testgen_with_float_param!();
                })*
                $(mod [<$int _ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<FloatType, $int, BoolType>;
                    pub type TestTensor<const D: usize> = TestTensor2<FloatType, $int, BoolType, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<FloatType, $int, BoolType, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<FloatType, $int, BoolType, D>;

                    pub type IntType = $int;

                    $crate::testgen_with_int_param!();
                })*
                $(mod [<$bool _bool_ty>] {
                    pub use super::*;

                    pub type TestBackend = TestBackend2<FloatType, IntType, $bool>;
                    pub type TestTensor<const D: usize> = TestTensor2<FloatType, IntType, $bool, D>;
                    pub type TestTensorInt<const D: usize> = TestTensorInt2<FloatType, IntType, $bool, D>;
                    pub type TestTensorBool<const D: usize> = TestTensorBool2<FloatType, IntType, $bool, D>;

                    pub type BoolType = $bool;

                    $crate::testgen_with_int_param!();
                })*
            }
            $crate::testgen_no_param!();
        }
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_quantization {
    () => {
        // Quantized tensor utilities
        pub mod qtensor {
            use core::marker::PhantomData;

            use burn_tensor::{
                backend::Backend,
                quantization::{QuantizationMode, QuantizationScheme, QuantizationType},
                Tensor, TensorData,
            };

            pub struct QTensor<B: Backend, const D: usize> {
                b: PhantomData<B>,
            }

            impl<B: Backend, const D: usize> QTensor<B, D> {
                /// Creates a quantized int8 tensor from the floating point data using the default quantization scheme
                /// (i.e., per-tensor symmetric quantization).
                pub fn int8<F: Into<TensorData>>(floats: F) -> Tensor<B, D> {
                    Self::int8_symmetric(floats)
                }
                /// Creates a quantized int8 tensor from the floating point data using per-tensor symmetric quantization.
                pub fn int8_symmetric<F: Into<TensorData>>(floats: F) -> Tensor<B, D> {
                    Tensor::from_floats(floats, &Default::default()).quantize_dynamic(
                        &QuantizationScheme::PerTensor(
                            QuantizationMode::Symmetric,
                            QuantizationType::QInt8,
                        ),
                    )
                }
                /// Creates a quantized int8 tensor from the floating point data using per-tensor affine quantization.
                pub fn int8_affine<F: Into<TensorData>>(floats: F) -> Tensor<B, D> {
                    Tensor::from_floats(floats, &Default::default()).quantize_dynamic(
                        &QuantizationScheme::PerTensor(
                            QuantizationMode::Affine,
                            QuantizationType::QInt8,
                        ),
                    )
                }
            }
        }
        pub use qtensor::*;

        // test quantization
        burn_tensor::testgen_calibration!();
        burn_tensor::testgen_scheme!();
        burn_tensor::testgen_quantize!();

        // test ops
        burn_tensor::testgen_q_abs!();
        burn_tensor::testgen_q_add!();
        burn_tensor::testgen_q_aggregation!();
        burn_tensor::testgen_q_all!();
        burn_tensor::testgen_q_any!();
        burn_tensor::testgen_q_arg!();
        burn_tensor::testgen_q_cat!();
        burn_tensor::testgen_q_chunk!();
        burn_tensor::testgen_q_clamp!();
        burn_tensor::testgen_q_cos!();
        burn_tensor::testgen_q_div!();
        burn_tensor::testgen_q_erf!();
        burn_tensor::testgen_q_exp!();
        burn_tensor::testgen_q_expand!();
        burn_tensor::testgen_q_flip!();
        burn_tensor::testgen_q_gather_scatter!();
        burn_tensor::testgen_q_log!();
        burn_tensor::testgen_q_log1p!();
        burn_tensor::testgen_q_map_comparison!();
        burn_tensor::testgen_q_mask!();
        burn_tensor::testgen_q_matmul!();
        burn_tensor::testgen_q_maxmin!();
        burn_tensor::testgen_q_mul!();
        burn_tensor::testgen_q_narrow!();
        burn_tensor::testgen_q_neg!();
        burn_tensor::testgen_q_permute!();
        burn_tensor::testgen_q_powf_scalar!();
        burn_tensor::testgen_q_powf!();
        burn_tensor::testgen_q_recip!();
        burn_tensor::testgen_q_remainder!();
        burn_tensor::testgen_q_repeat_dim!();
        burn_tensor::testgen_q_reshape!();
        burn_tensor::testgen_q_round!();
        burn_tensor::testgen_q_select!();
        burn_tensor::testgen_q_sin!();
        burn_tensor::testgen_q_slice!();
        burn_tensor::testgen_q_sort_argsort!();
        burn_tensor::testgen_q_split!();
        burn_tensor::testgen_q_sqrt!();
        burn_tensor::testgen_q_stack!();
        burn_tensor::testgen_q_sub!();
        burn_tensor::testgen_q_tanh!();
        burn_tensor::testgen_q_topk!();
        burn_tensor::testgen_q_transpose!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_with_float_param {
    () => {
        // test activation
        burn_tensor::testgen_gelu!();
        burn_tensor::testgen_mish!();
        burn_tensor::testgen_relu!();
        burn_tensor::testgen_leaky_relu!();
        burn_tensor::testgen_softmax!();
        burn_tensor::testgen_softmin!();
        burn_tensor::testgen_softplus!();
        burn_tensor::testgen_sigmoid!();
        burn_tensor::testgen_log_sigmoid!();
        burn_tensor::testgen_silu!();
        burn_tensor::testgen_tanh_activation!();

        // test module
        burn_tensor::testgen_module_conv1d!();
        burn_tensor::testgen_module_conv2d!();
        burn_tensor::testgen_module_conv3d!();
        burn_tensor::testgen_module_forward!();
        burn_tensor::testgen_module_deform_conv2d!();
        burn_tensor::testgen_module_conv_transpose1d!();
        burn_tensor::testgen_module_conv_transpose2d!();
        burn_tensor::testgen_module_conv_transpose3d!();
        burn_tensor::testgen_module_unfold4d!();
        burn_tensor::testgen_module_max_pool1d!();
        burn_tensor::testgen_module_max_pool2d!();
        burn_tensor::testgen_module_avg_pool1d!();
        burn_tensor::testgen_module_avg_pool2d!();
        burn_tensor::testgen_module_adaptive_avg_pool1d!();
        burn_tensor::testgen_module_adaptive_avg_pool2d!();
        burn_tensor::testgen_module_nearest_interpolate!();
        burn_tensor::testgen_module_bilinear_interpolate!();
        burn_tensor::testgen_module_bicubic_interpolate!();

        // test ops
        burn_tensor::testgen_gather_scatter!();
        burn_tensor::testgen_narrow!();
        burn_tensor::testgen_add!();
        burn_tensor::testgen_aggregation!();
        burn_tensor::testgen_arange!();
        burn_tensor::testgen_arange_step!();
        burn_tensor::testgen_arg!();
        burn_tensor::testgen_cast!();
        burn_tensor::testgen_cat!();
        burn_tensor::testgen_chunk!();
        burn_tensor::testgen_clamp!();
        burn_tensor::testgen_close!();
        burn_tensor::testgen_cos!();
        burn_tensor::testgen_create_like!();
        burn_tensor::testgen_div!();
        burn_tensor::testgen_erf!();
        burn_tensor::testgen_exp!();
        burn_tensor::testgen_flatten!();
        burn_tensor::testgen_full!();
        burn_tensor::testgen_init!();
        burn_tensor::testgen_iter_dim!();
        burn_tensor::testgen_log!();
        burn_tensor::testgen_log1p!();
        burn_tensor::testgen_map_comparison!();
        burn_tensor::testgen_mask!();
        burn_tensor::testgen_matmul!();
        burn_tensor::testgen_maxmin!();
        burn_tensor::testgen_mul!();
        burn_tensor::testgen_neg!();
        burn_tensor::testgen_one_hot!();
        burn_tensor::testgen_powf_scalar!();
        burn_tensor::testgen_random!();
        burn_tensor::testgen_recip!();
        burn_tensor::testgen_repeat_dim!();
        burn_tensor::testgen_repeat!();
        burn_tensor::testgen_reshape!();
        burn_tensor::testgen_sin!();
        burn_tensor::testgen_slice!();
        burn_tensor::testgen_stack!();
        burn_tensor::testgen_sqrt!();
        burn_tensor::testgen_abs!();
        burn_tensor::testgen_squeeze!();
        burn_tensor::testgen_sub!();
        burn_tensor::testgen_tanh!();
        burn_tensor::testgen_transpose!();
        burn_tensor::testgen_tri!();
        burn_tensor::testgen_powf!();
        burn_tensor::testgen_any!();
        burn_tensor::testgen_all_op!();
        burn_tensor::testgen_permute!();
        burn_tensor::testgen_movedim!();
        burn_tensor::testgen_flip!();
        burn_tensor::testgen_bool!();
        burn_tensor::testgen_argwhere_nonzero!();
        burn_tensor::testgen_sign!();
        burn_tensor::testgen_expand!();
        burn_tensor::testgen_tri_mask!();
        burn_tensor::testgen_sort_argsort!();
        burn_tensor::testgen_topk!();
        burn_tensor::testgen_remainder!();
        burn_tensor::testgen_cartesian_grid!();
        burn_tensor::testgen_nan!();
        burn_tensor::testgen_round!();
        burn_tensor::testgen_floor!();
        burn_tensor::testgen_ceil!();
        burn_tensor::testgen_select!();
        burn_tensor::testgen_split!();
        burn_tensor::testgen_prod!();

        // test stats
        burn_tensor::testgen_var!();
        burn_tensor::testgen_cov!();
        burn_tensor::testgen_eye!();

        // test padding
        burn_tensor::testgen_padding!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_with_int_param {
    () => {
        // test ops
        burn_tensor::testgen_add!();
        burn_tensor::testgen_aggregation!();
        burn_tensor::testgen_arg!();
        burn_tensor::testgen_cast!();
        burn_tensor::testgen_bool!();
        burn_tensor::testgen_cat!();
        burn_tensor::testgen_div!();
        burn_tensor::testgen_expand!();
        burn_tensor::testgen_flip!();
        burn_tensor::testgen_mask!();
        burn_tensor::testgen_movedim!();
        burn_tensor::testgen_mul!();
        burn_tensor::testgen_permute!();
        burn_tensor::testgen_reshape!();
        burn_tensor::testgen_select!();
        burn_tensor::testgen_sign!();
        burn_tensor::testgen_sort_argsort!();
        burn_tensor::testgen_stack!();
        burn_tensor::testgen_sub!();
        burn_tensor::testgen_transpose!();
        burn_tensor::testgen_gather_scatter!();
        burn_tensor::testgen_bitwise!();

        // test stats
        burn_tensor::testgen_eye!();

        // test padding
        burn_tensor::testgen_padding!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_with_bool_param {
    () => {
        burn_tensor::testgen_all_op!();
        burn_tensor::testgen_any_op!();
        burn_tensor::testgen_argwhere_nonzero!();
        burn_tensor::testgen_cast!();
        burn_tensor::testgen_cat!();
        burn_tensor::testgen_expand!();
        burn_tensor::testgen_full!();
        burn_tensor::testgen_map_comparison!();
        burn_tensor::testgen_mask!();
        burn_tensor::testgen_nan!();
        burn_tensor::testgen_repeat_dim!();
        burn_tensor::testgen_repeat!();
        burn_tensor::testgen_reshape!();
        burn_tensor::testgen_stack!();
        burn_tensor::testgen_transpose!();
        burn_tensor::tri_mask!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_no_param {
    () => {
        // test stats
        burn_tensor::testgen_display!();

        // test clone invariance
        burn_tensor::testgen_clone_invariance!();

        // test primitive
        burn_tensor::testgen_primitive!();
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_bytes {
    ($ty:ident: $($elem:expr),*) => {
        F::as_bytes(&[$($ty::new($elem),)*])
    };
}

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_type {
    ($ty:ident: [$($elem:tt),*]) => {
        [$($crate::as_type![$ty: $elem]),*]
    };
    ($ty:ident: [$($elem:tt,)*]) => {
        [$($crate::as_type![$ty: $elem]),*]
    };
    ($ty:ident: $elem:expr) => {
        {
            use $crate::tests::{Float, Int};

            $ty::new($elem)
        }
    };
}
