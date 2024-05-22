use crate::dialect::{Elem, FloatKind, Variable};
use crate::language::{CubeContext, CubeType, ExpandElement, Numeric, CubeElem};
use crate::unexpanded;

/// Floating point numbers. Used as input in float kernels
pub trait Float: Numeric {
    fn new(val: f64) -> Self;
    fn new_expand(context: &mut CubeContext, val: f64) -> <Self as CubeType>::ExpandType;
}

macro_rules! impl_float {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: f64,
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
            fn new(_val: f64) -> Self {
                unexpanded!()
            }

            fn new_expand(_context: &mut CubeContext, val: f64) -> <Self as CubeType>::ExpandType {
                let new_var = Variable::ConstantScalar(val, Self::as_elem());
                ExpandElement::Plain(new_var)
            }
        }
    };
}

impl_float!(F16);
impl_float!(BF16);
impl_float!(F32);
impl_float!(F64);
