use crate::dialect::{Elem, FloatKind, Variable};
use crate::{CubeContext, CubeType, ExpandElement, Numeric, PrimitiveVariable};
use std::rc::Rc;

/// Floating point numbers. Used as input in float kernels
pub trait Float: Numeric {
    fn new(val: f64) -> Self;
    fn new_expand(context: &mut CubeContext, val: f64) -> <Self as CubeType>::ExpandType;
}

macro_rules! impl_float {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: <Self as PrimitiveVariable>::Primitive,
            pub vectorization: usize,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElement;
        }

        impl PrimitiveVariable for $type {
            type Primitive = f64;

            /// Return the element type to use on GPU
            fn into_elem() -> Elem {
                Elem::Float(FloatKind::$type)
            }

            fn to_f64(&self) -> f64 {
                self.val
            }

            fn from_f64(val: f64) -> Self {
                Self::new(val)
            }

            fn from_i64(val: i64) -> Self {
                Self::new(val as f64)
            }
        }

        impl Numeric for $type {}

        impl Float for $type {
            fn new(val: <Self as PrimitiveVariable>::Primitive) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }

            fn new_expand(
                _context: &mut CubeContext,
                val: <Self as PrimitiveVariable>::Primitive,
            ) -> <Self as CubeType>::ExpandType {
                let new_var = Variable::ConstantScalar(val as f64, Self::into_elem());
                ExpandElement::new(Rc::new(new_var))
            }
        }
    };
}

impl_float!(F16);
impl_float!(BF16);
impl_float!(F32);
impl_float!(F64);
