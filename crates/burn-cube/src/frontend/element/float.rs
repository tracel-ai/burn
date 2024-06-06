use half::{bf16, f16};

use crate::frontend::{Ceil, Cos, Erf, Exp, Floor, Log, Log1p, Powf, Recip, Sin, Sqrt, Tanh};
use crate::frontend::{CubeContext, CubeElem, CubeType, ExpandElement, Numeric};
use crate::ir::{Elem, FloatKind, Item, Variable, Vectorization};

use crate::compute::{KernelBuilder, KernelLauncher};
use crate::prelude::index_assign;
use crate::{unexpanded, Runtime};

use super::{ArgSettings, LaunchArg, UInt, Vectorized};

/// Floating point numbers. Used as input in float kernels
pub trait Float:
    Numeric
    + Exp
    + Log
    + Log1p
    + Cos
    + Sin
    + Tanh
    + Powf
    + Sqrt
    + Floor
    + Ceil
    + Erf
    + Recip
    + core::ops::Index<UInt, Output = Self>
{
    fn new(val: f32) -> Self;
    fn new_expand(context: &mut CubeContext, val: f32) -> <Self as CubeType>::ExpandType;
    fn vectorized(val: f32, vectorization: UInt) -> Self;
    fn vectorized_expand(
        context: &mut CubeContext,
        val: f32,
        vectorization: UInt,
    ) -> <Self as CubeType>::ExpandType;
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

            fn vectorized(val: f32, vectorization: UInt) -> Self {
                if vectorization.val == 1 {
                    Self::new(val)
                } else {
                    Self {
                        val,
                        vectorization: vectorization.val as u8,
                    }
                }
            }

            fn vectorized_expand(
                context: &mut CubeContext,
                val: f32,
                vectorization: UInt,
            ) -> <Self as CubeType>::ExpandType {
                if vectorization.val == 1 {
                    Self::new_expand(context, val)
                } else {
                    let mut new_var = context
                        .create_local(Item::vectorized(Self::as_elem(), vectorization.val as u8));
                    for (i, element) in vec![val; vectorization.val as usize].iter().enumerate() {
                        new_var = index_assign::expand(context, new_var, i, *element);
                    }

                    new_var
                }
            }
        }

        impl core::ops::Index<UInt> for $type {
            type Output = Self;

            fn index(&self, _index: UInt) -> &Self::Output {
                unexpanded!()
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

        impl Vectorized for $type {
            fn vectorization_factor(&self) -> UInt {
                UInt {
                    val: self.vectorization as u32,
                    vectorization: 1,
                }
            }

            fn vectorize(mut self, factor: UInt) -> Self {
                self.vectorization = factor.vectorization;
                self
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
