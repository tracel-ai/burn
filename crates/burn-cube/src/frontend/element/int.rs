use crate::compute::{KernelBuilder, KernelLauncher};
use crate::frontend::{CubeContext, CubeElem, CubeType, ExpandElement, Numeric};
use crate::ir::{Elem, IntKind, Variable, Vectorization};
use crate::Runtime;

use super::{ArgSettings, LaunchArg};

/// Signed integer. Used as input in int kernels
pub trait Int: Numeric + std::ops::Rem<Output = Self> {
    fn new(val: i64) -> Self;
    fn new_expand(context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType;
}

macro_rules! impl_int {
    ($type:ident, $primitive:ty) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: i64,
            pub vectorization: u8,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElement;
        }

        impl CubeElem for $type {
            fn as_elem() -> Elem {
                Elem::Int(IntKind::$type)
            }
        }

        impl Numeric for $type {}

        impl Int for $type {
            fn new(val: i64) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }

            fn new_expand(_context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType {
                let new_var = Variable::ConstantScalar(val as f64, Self::as_elem());
                ExpandElement::Plain(new_var)
            }
        }

        impl LaunchArg for $type {
            type RuntimeArg<'a, R: Runtime> = $primitive;

            fn compile_input(
                builder: &mut KernelBuilder,
                vectorization: Vectorization,
            ) -> ExpandElement {
                assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
                builder.scalar(Self::as_elem())
            }

            fn compile_output(
                builder: &mut KernelBuilder,
                vectorization: Vectorization,
            ) -> ExpandElement {
                assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
                builder.scalar(Self::as_elem())
            }
        }
    };
}

impl_int!(I32, i32);
impl_int!(I64, i64);

impl<R: Runtime> ArgSettings<R> for i32 {
    fn register(&self, settings: &mut KernelLauncher<R>) {
        settings.register_i32(*self);
    }
}

impl<R: Runtime> ArgSettings<R> for i64 {
    fn register(&self, settings: &mut KernelLauncher<R>) {
        settings.register_i64(*self);
    }
}
