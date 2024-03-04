#![allow(missing_docs)]

mod reduce;

#[cfg(feature = "export_tests")]
#[macro_export]
macro_rules! testgen_all {
    () => {
        mod jit {
            burn_jit::testgen_jit!();
            burn_jit::testgen_reduction!();
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
