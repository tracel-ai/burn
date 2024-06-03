use half::{bf16, f16};

use crate::frontend::{Ceil, Cos, Erf, Exp, Floor, Log, Log1p, Powf, Recip, Sin, Sqrt, Tanh};
use crate::frontend::{CubeContext, CubeElem, CubeType, ExpandElement, Numeric};
use crate::ir::{Elem, FloatKind, Variable, Vectorization};

use crate::compute::{KernelBuilder, KernelLauncher};
use crate::Runtime;

use super::{ArgSettings, LaunchArg};

/// Floating point numbers. Used as input in float kernels
pub trait Float:
    Numeric + Exp + Log + Log1p + Cos + Sin + Tanh + Powf + Sqrt + Floor + Ceil + Erf + Recip
{
    fn new(val: f32) -> Self;
    fn new_expand(context: &mut CubeContext, val: f32) -> <Self as CubeType>::ExpandType;
}

macro_rules! impl_float {
    ($type:ident, $primitive:ty) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: f32,
            pub vectorization: u8,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElement;
        }

        impl CubeElem for $type {
            /// Return the element type to use on GPU
            fn as_elem() -> Elem {
                Elem::Float(FloatKind::$type)
            }
        }

        impl Numeric for $type {}

        impl Float for $type {
            fn new(val: f32) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }

            fn new_expand(_context: &mut CubeContext, val: f32) -> <Self as CubeType>::ExpandType {
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

impl_float!(F16, f16);
impl_float!(BF16, bf16);
impl_float!(F32, f32);
impl_float!(F64, f64);

impl<R: Runtime> ArgSettings<R> for f16 {
    fn register(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f16(*self);
    }
}

impl<R: Runtime> ArgSettings<R> for bf16 {
    fn register(&self, settings: &mut KernelLauncher<R>) {
        settings.register_bf16(*self);
    }
}

impl<R: Runtime> ArgSettings<R> for f32 {
    fn register(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32(*self);
    }
}

impl<R: Runtime> ArgSettings<R> for f64 {
    fn register(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f64(*self);
    }
}
