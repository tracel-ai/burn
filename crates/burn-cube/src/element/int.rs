use crate::{CubeContext, CubeType, ExpandElement, Numeric};
use burn_jit::gpu::{Elem, IntKind, Variable};
use std::rc::Rc;

pub trait Int:
    Clone
    + Copy
    + std::cmp::PartialOrd
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + Numeric
{
    fn into_elem() -> Elem;
    fn from(val: i64) -> Self;
    fn from_expand(context: &mut CubeContext, val: i64) -> ExpandElement;
}

macro_rules! impl_int {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: i64,
            pub vectorization: usize,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElement;
        }

        impl Int for $type {
            fn into_elem() -> Elem {
                Elem::Int(IntKind::$type)
            }
            fn from(val: i64) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }
            fn from_expand(_context: &mut CubeContext, val: i64) -> ExpandElement {
                let new_var = Variable::ConstantScalar(val as f64, Self::into_elem());
                ExpandElement::new(Rc::new(new_var))
            }
        }

        impl Numeric for $type {
            fn new(val: i64) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }
            fn new_expand(context: &mut CubeContext, val: i64) -> ExpandElement {
                <Self as Int>::from_expand(context, val)
            }
        }
    };
}

impl_int!(I32);
impl_int!(I64);
