#![allow(missing_docs)]

mod conv2d;
mod conv_transpose2d;
mod gather;
mod reduce;
mod repeat;
mod scatter;
mod select;
mod select_assign;
mod slice;
mod slice_assign;

#[cfg(feature = "export_tests")]
#[macro_export]
macro_rules! testgen_all {
    () => {
        mod jit {
            burn_jit::testgen_jit!();

            mod kernel {
                use super::*;

                burn_jit::testgen_reduction!();
                burn_jit::testgen_conv2d!();
                burn_jit::testgen_conv_transpose2d!();
                burn_jit::testgen_gather!();
                burn_jit::testgen_repeat!();
                burn_jit::testgen_scatter!();
                burn_jit::testgen_select!();
                burn_jit::testgen_select_assign!();
                burn_jit::testgen_slice!();
                burn_jit::testgen_slice_assign!();
            }
        }
        mod jit_fusion {
            burn_jit::testgen_jit_fusion!();
        }
    };
}

#[cfg(feature = "export_tests")]
#[macro_export]
macro_rules! testgen_jit {
    () => {
        use super::*;

        pub type TestBackend = JitBackend<TestRuntime>;
        pub type ReferenceBackend = burn_ndarray::NdArray<f32>;

        pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
        pub type TestTensorInt<const D: usize> =
            burn_tensor::Tensor<TestBackend, D, burn_tensor::Int>;
        pub type TestTensorBool<const D: usize> =
            burn_tensor::Tensor<TestBackend, D, burn_tensor::Bool>;

        pub type ReferenceTensor<const D: usize> = burn_tensor::Tensor<ReferenceBackend, D>;

        burn_tensor::testgen_all!();
        burn_autodiff::testgen_all!();
    };
}
