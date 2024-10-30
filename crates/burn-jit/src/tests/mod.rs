#![allow(missing_docs)]

mod avg_pool2d;
mod bernoulli;
mod cast;
mod cat;
mod clamp;
mod conv2d;
mod conv3d;
mod conv_transpose2d;
mod conv_transpose3d;
mod gather;
mod mask_fill;
mod mask_where;
mod matmul;
mod max_pool2d;
mod max_pool2d_backward;
mod normal;
mod quantization;
mod reduce;
mod repeat_dim;
mod scatter;
mod select;
mod select_assign;
mod slice;
mod slice_assign;
mod unary;
mod uniform;

// Re-export dependencies for tests
pub use burn_autodiff;
pub use burn_fusion;
pub use burn_ndarray;
pub use burn_tensor;
pub use serial_test;

#[macro_export]
macro_rules! testgen_all {
    () => {
        $crate::testgen_all!(f32: [], i32: []);
    };
    ($f_def:ident: [$($float:ident),*], $i_def:ident: [$($int:ident),*]) => {
        mod jit {
            burn_jit::testgen_jit!($f_def: [$($float),*], $i_def: [$($int),*]);

            mod kernel {
                use super::*;

                burn_jit::testgen_reduction!();
                burn_jit::testgen_conv2d!();
                burn_jit::testgen_conv3d!();
                burn_jit::testgen_conv_transpose2d!();
                burn_jit::testgen_conv_transpose3d!();

                burn_jit::testgen_repeat_dim!();
                burn_jit::testgen_gather!();
                burn_jit::testgen_scatter!();

                burn_jit::testgen_select!();
                burn_jit::testgen_select_assign!();

                burn_jit::testgen_slice!();
                burn_jit::testgen_slice_assign!();

                burn_jit::testgen_mask_where!();
                burn_jit::testgen_mask_fill!();

                burn_jit::testgen_avg_pool2d!();
                burn_jit::testgen_max_pool2d!();
                burn_jit::testgen_max_pool2d_backward!();

                burn_jit::testgen_bernoulli!();
                burn_jit::testgen_normal!();
                burn_jit::testgen_uniform!();

                burn_jit::testgen_cast!();
                burn_jit::testgen_cat!();
                burn_jit::testgen_clamp!();
                burn_jit::testgen_unary!();

                burn_jit::testgen_quantization!();
            }
        }
        mod jit_fusion {
            burn_jit::testgen_jit_fusion!($f_def: [$($float),*], $i_def: [$($int),*]);
        }
    };
}

#[macro_export]
macro_rules! testgen_jit {
    () => {
        $crate::testgen_jit!(f32: [], i32: []);
    };
    ($f_def:ident: [$($float:ident),*], $i_def:ident: [$($int:ident),*]) => {
        pub use super::*;
        use burn_jit::tests::{burn_autodiff, burn_ndarray, burn_tensor, serial_test};

        pub type TestBackend = JitBackend<TestRuntime, f32, i32>;
        pub type TestBackend2<F, I> = JitBackend<TestRuntime, F, I>;
        pub type ReferenceBackend = burn_ndarray::NdArray<f32>;

        pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
        pub type TestTensor2<F, I, const D: usize> = burn_tensor::Tensor<TestBackend2<F, I>, D>;
        pub type TestTensorInt<const D: usize> =
            burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;
        pub type TestTensorInt2<F, I, const D: usize> =
            burn_tensor::Tensor<TestBackend2<F, I>, D, burn_tensor::Int>;
        pub type TestTensorBool<const D: usize> =
            burn_tensor::Tensor<TestBackend, D, burn_tensor::Bool>;
        pub type TestTensorBool2<F, I, const D: usize> =
            burn_tensor::Tensor<TestBackend2<F, I>, D, burn_tensor::Bool>;

        pub type ReferenceTensor<const D: usize> = burn_tensor::Tensor<ReferenceBackend, D>;

        burn_tensor::testgen_all!($f_def: [$($float),*], $i_def: [$($int),*]);
        burn_autodiff::testgen_all!($f_def: [$($float),*], $i_def: [$($int),*]);

        // Not all ops are implemented for quantization yet, notably missing:
        // `q_swap_dims`, `q_permute`, `q_flip`, `q_gather`, `q_select`, `q_slice`, `q_expand`
        // burn_tensor::testgen_quantization!();
        // test quantization
        burn_tensor::testgen_calibration!();
        burn_tensor::testgen_scheme!();
        burn_tensor::testgen_quantize!();
    }
}

#[macro_export]
macro_rules! testgen_jit_fusion {
    () => {
        $crate::testgen_jit_fusion!(f32: [], i32: []);
    };
    ($f_def:ident: [$($float:ident),*], $i_def:ident: [$($int:ident),*]) => {
        use super::*;
        use burn_jit::tests::{burn_autodiff, burn_fusion, burn_ndarray, burn_tensor};

        pub type TestBackend = burn_fusion::Fusion<JitBackend<TestRuntime, f32, i32>>;
        pub type TestBackend2<F, I> = burn_fusion::Fusion<JitBackend<TestRuntime, F, I>>;
        pub type ReferenceBackend = burn_ndarray::NdArray<f32>;

        pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
        pub type TestTensor2<F, I, const D: usize> = burn_tensor::Tensor<TestBackend2<F, I>, D>;
        pub type TestTensorInt<const D: usize> =
            burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;
        pub type TestTensorInt2<F, I, const D: usize> =
            burn_tensor::Tensor<TestBackend2<F, I>, D, burn_tensor::Int>;
        pub type TestTensorBool<const D: usize> =
            burn_tensor::Tensor<TestBackend, D, burn_tensor::Bool>;
        pub type TestTensorBool2<F, I, const D: usize> =
            burn_tensor::Tensor<TestBackend2<F, I>, D, burn_tensor::Bool>;

        pub type ReferenceTensor<const D: usize> = burn_tensor::Tensor<ReferenceBackend, D>;

        burn_tensor::testgen_all!($f_def: [$($float),*], $i_def: [$($int),*]);
        burn_autodiff::testgen_all!($f_def: [$($float),*], $i_def: [$($int),*]);

        // Not all ops are implemented for quantization yet, notably missing:
        // `q_swap_dims`, `q_permute`, `q_flip`, `q_gather`, `q_select`, `q_slice`, `q_expand`
        // burn_tensor::testgen_quantization!();
        // test quantization
        burn_tensor::testgen_calibration!();
        burn_tensor::testgen_scheme!();
        burn_tensor::testgen_quantize!();
    };
}
