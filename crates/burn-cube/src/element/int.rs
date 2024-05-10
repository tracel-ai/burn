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
    fn into_kind() -> IntKind;
}

macro_rules! impl_int {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: f64,
            pub vectorization: usize,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElement;
        }

        impl Int for $type {
            fn into_kind() -> IntKind {
                IntKind::$type
            }
        }

        impl Numeric for $type {
            fn new(val: f64) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }
            fn new_expand(_context: &mut CubeContext, val: f64) -> ExpandElement {
                let elem = Elem::Int(Self::into_kind());
                let new_var = Variable::ConstantScalar(val, elem);
                ExpandElement::new(Rc::new(new_var))
            }
        }
    };
}

impl_int!(I32);
impl_int!(I64);
