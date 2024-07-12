use half::{bf16, f16};

use crate::frontend::{Ceil, Cos, Erf, Exp, Floor, Log, Log1p, Powf, Recip, Sin, Sqrt, Tanh};
use crate::frontend::{CubeContext, CubePrimitive, CubeType, ExpandElement, Numeric};
use crate::ir::{Elem, FloatKind, Item, Variable, Vectorization};

use crate::compute::{KernelBuilder, KernelLauncher};
use crate::prelude::index_assign;
use crate::{unexpanded, Runtime};

use super::{LaunchArgExpand, ScalarArgSettings, UInt, Vectorized};

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
    + core::ops::IndexMut<UInt, Output = Self>
{
    fn new(val: f32) -> Self;
    fn vectorized(val: f32, vectorization: UInt) -> Self;
    fn vectorized_empty(vectorization: UInt) -> Self;
    fn __expand_new(context: &mut CubeContext, val: f32) -> <Self as CubeType>::ExpandType;
    fn __expand_vectorized(
        context: &mut CubeContext,
        val: f32,
        vectorization: UInt,
    ) -> <Self as CubeType>::ExpandType;
    fn __expand_vectorized_empty(
        context: &mut CubeContext,
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

        impl CubePrimitive for $type {
            /// Return the element type to use on GPU
            fn as_elem() -> Elem {
                Elem::Float(FloatKind::$type)
            }
        }

        impl Numeric for $type {
            type Primitive = $primitive;
        }

        impl Float for $type {
            fn new(val: f32) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
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

            fn vectorized_empty(vectorization: UInt) -> Self {
                Self::vectorized(0., vectorization)
            }

            fn __expand_new(
                _context: &mut CubeContext,
                val: f32,
            ) -> <Self as CubeType>::ExpandType {
                let new_var = Variable::ConstantScalar {
                    value: val as f64,
                    elem: Self::as_elem(),
                };
                ExpandElement::Plain(new_var)
            }

            fn __expand_vectorized(
                context: &mut CubeContext,
                val: f32,
                vectorization: UInt,
            ) -> <Self as CubeType>::ExpandType {
                if vectorization.val == 1 {
                    Self::__expand_new(context, val)
                } else {
                    let mut new_var = context
                        .create_local(Item::vectorized(Self::as_elem(), vectorization.val as u8));
                    for (i, element) in vec![val; vectorization.val as usize].iter().enumerate() {
                        new_var = index_assign::expand(context, new_var, i, *element);
                    }

                    new_var
                }
            }

            fn __expand_vectorized_empty(
                context: &mut CubeContext,
                vectorization: UInt,
            ) -> <Self as CubeType>::ExpandType {
                if vectorization.val == 1 {
                    Self::__expand_new(context, 0.)
                } else {
                    context.create_local(Item::vectorized(Self::as_elem(), vectorization.val as u8))
                }
            }
        }

        impl core::ops::Index<UInt> for $type {
            type Output = Self;

            fn index(&self, _index: UInt) -> &Self::Output {
                unexpanded!()
            }
        }

        impl core::ops::IndexMut<UInt> for $type {
            fn index_mut(&mut self, _index: UInt) -> &mut Self::Output {
                unexpanded!()
            }
        }

        impl LaunchArgExpand for $type {
            fn expand(builder: &mut KernelBuilder, vectorization: Vectorization) -> ExpandElement {
                assert_eq!(vectorization, 1, "Attempted to vectorize a scalar");
                builder.scalar($type::as_elem())
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

impl From<f32> for F32 {
    fn from(value: f32) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

impl From<f32> for BF16 {
    fn from(value: f32) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

impl From<f32> for F16 {
    fn from(value: f32) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

impl From<f32> for F64 {
    fn from(value: f32) -> Self {
        Self {
            val: value,
            vectorization: 1,
        }
    }
}

impl ScalarArgSettings for f16 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f16(*self);
    }
}

impl ScalarArgSettings for bf16 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_bf16(*self);
    }
}

impl ScalarArgSettings for f32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32(*self);
    }
}

impl ScalarArgSettings for f64 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f64(*self);
    }
}
