use crate::{ExpandElement, RuntimeType};
use burn_jit::gpu::{Elem, FloatKind, Variable};
use std::rc::Rc;

pub trait Float:
    Clone
    + Copy
    + RuntimeType<ExpandType = ExpandElement>
    + std::cmp::PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn into_kind() -> FloatKind;
    fn new(val: f32, vectorization: usize) -> Self;
    fn new_expand(val: f32) -> ExpandElement;
}

macro_rules! impl_float {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: f32,
            pub vectorization: usize,
        }

        impl RuntimeType for $type {
            type ExpandType = ExpandElement;
        }

        impl Float for $type {
            fn into_kind() -> FloatKind {
                FloatKind::$type
            }
            fn new(val: f32, vectorization: usize) -> Self {
                Self { val, vectorization }
            }
            fn new_expand(val: f32) -> ExpandElement {
                let elem = Elem::Float(Self::into_kind());
                let new_var = Variable::ConstantScalar(val as f64, elem);
                ExpandElement::new(Rc::new(new_var))
            }
        }
    };
}

impl_float!(F16);
impl_float!(BF16);
impl_float!(F32);
impl_float!(F64);
