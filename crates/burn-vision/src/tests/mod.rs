mod connected_components;

#[macro_export]
macro_rules! testgen_all {
    () => {
        use burn_tensor::{Bool, Float, Int};

        pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, Int>;
        pub type TestTensorBool<const D: usize> = burn_tensor::Tensor<TestBackend, D, Bool>;

        pub mod vision {
            pub use super::*;

            pub type IntType = <TestBackend as burn_tensor::backend::Backend>::IntElem;

            burn_vision::testgen_connected_components!();
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
            use cubecl::prelude::*;

            $ty::new($elem)
        }
    };
}
