use crate::{CubeContext, CubeType, ExpandElement, Numeric, PrimitiveVariable};
use burn_jit::gpu::{Elem, FloatKind, Variable};
use std::rc::Rc;

/// Floating point numbers. Used as input in float kernels
pub trait Float: Numeric {
    /// Create a Float from a float literal
    fn from_primitive(val: f64) -> Self;
    /// Expand version of from_primitive
    fn from_primitive_expand(context: &mut CubeContext, val: f64)
        -> <Self as CubeType>::ExpandType;
}

macro_rules! impl_float {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: f64,
            pub vectorization: usize,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElement;
        }

        impl PrimitiveVariable for $type {
            /// Note: all float types have f64 primitive on CPU to
            /// ease casting. On GPU the type will be given by into_elem.
            type Primitive = f64;

            /// Return the value of the float (on CPU)
            fn val(&self) -> Self::Primitive {
                self.val
            }

            /// Return the element type to use on GPU
            fn into_elem() -> Elem {
                Elem::Float(FloatKind::$type)
            }
        }

        impl Float for $type {
            fn from_primitive(val: f64) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }

            fn from_primitive_expand(
                _context: &mut CubeContext,
                val: f64,
            ) -> <Self as CubeType>::ExpandType {
                let new_var = Variable::ConstantScalar(val, Self::into_elem());
                ExpandElement::new(Rc::new(new_var))
            }
        }

        impl Numeric for $type {
            // Method new takes an i64, because it is used when treating the float as numeric,
            // which must be an int in the cube kernel because new numerics need to be supported by Int as well
            fn lit(val: i64) -> Self {
                Self::from_primitive(val as f64)
            }

            fn lit_expand(context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType {
                <Self as Float>::from_primitive_expand(context, val as f64)
            }
        }
    };
}

impl_float!(F16);
impl_float!(BF16);
impl_float!(F32);
impl_float!(F64);
