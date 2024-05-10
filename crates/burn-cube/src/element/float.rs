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
    fn into_kind() -> FloatKind;
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
            fn into_kind() -> FloatKind {
                FloatKind::$type
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
                let elem = Elem::Float(Self::into_kind());
                let new_var = Variable::ConstantScalar(val, elem);
                ExpandElement::new(Rc::new(new_var))
            }
        }
    };
}

impl_float!(F16);
impl_float!(BF16);
impl_float!(F32);
impl_float!(F64);
