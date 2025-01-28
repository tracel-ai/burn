mod connected_components;

#[macro_export]
macro_rules! testgen_all {
    () => {
        use burn_tensor::{Bool, Float, Int};

        pub type TestBackend = JitBackend<TestRuntime, f32, i32, u32>;

        type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
        type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, Int>;
        type TestTensorBool<const D: usize> = burn_tensor::Tensor<TestBackend, D, Bool>;

        pub mod vision {
            pub use super::*;

            pub type FloatType = <TestBackend as burn_tensor::backend::Backend>::FloatElem;
            pub type IntType = <TestBackend as burn_tensor::backend::Backend>::IntElem;
            pub type BoolType = <TestBackend as burn_tensor::backend::Backend>::BoolElem;

            $crate::testgen_connected_components!();
        }
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
            use cubecl::prelude::{Float, Int};

            $ty::new($elem)
        }
    };
}
