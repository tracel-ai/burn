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

pub(crate) mod codegen;
pub(crate) mod tune;

mod element;
pub use codegen::compiler::Compiler;
pub use codegen::dialect::gpu;

pub use element::{FloatElement, IntElement, JitElement};

mod backend;
pub use backend::*;
mod runtime;
pub use runtime::*;

#[cfg(any(feature = "fusion", test))]
mod fusion;

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
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
#[allow(missing_docs)]
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
