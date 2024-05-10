use crate::{CubeContext, CubeType, ExpandElement, Numeric};
use burn_jit::gpu::{Elem, FloatKind, Variable};
use std::rc::Rc;

pub trait Float:
    Clone
    + Copy
    + std::cmp::PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + Numeric
{
    fn from(val: f64) -> Self;
    fn from_expand(context: &mut CubeContext, val: f64) -> ExpandElement;
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

        impl Float for $type {
            fn from(val: f64) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }
            fn from_expand(_context: &mut CubeContext, val: f64) -> ExpandElement {
                let new_var = Variable::ConstantScalar(val, Self::into_elem());
                ExpandElement::new(Rc::new(new_var))
            }
        }

        impl Numeric for $type {
            fn new(val: i64) -> Self {
                Self {
                    val: val as f64,
                    vectorization: 1,
                }
            }
            fn new_expand(context: &mut CubeContext, val: i64) -> ExpandElement {
                <Self as Float>::from_expand(context, val as f64)
            }
            fn into_elem() -> Elem {
                Elem::Float(FloatKind::$type)
            }
        }
    };
}

impl_float!(F16);
impl_float!(BF16);
impl_float!(F32);
impl_float!(F64);
