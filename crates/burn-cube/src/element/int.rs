use crate::{ExpandElement, RuntimeType};
use burn_jit::gpu::{Elem, IntKind, Variable};
use std::rc::Rc;

pub trait Int:
    Clone
    + Copy
    + RuntimeType<ExpandType = ExpandElement>
    + std::cmp::PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
{
    fn into_kind() -> IntKind;
    fn new(val: i32, vectorization: usize) -> Self;
    fn new_expand(val: i32) -> ExpandElement;
}

macro_rules! impl_int {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: i32,
            pub vectorization: usize,
        }

        impl RuntimeType for $type {
            type ExpandType = ExpandElement;
        }

        impl Int for $type {
            fn into_kind() -> IntKind {
                IntKind::$type
            }
            fn new(val: i32, vectorization: usize) -> Self {
                Self { val, vectorization }
            }
            fn new_expand(val: i32) -> ExpandElement {
                let elem = Elem::Int(Self::into_kind());
                let new_var = Variable::ConstantScalar(val as f64, elem);
                ExpandElement::new(Rc::new(new_var))
            }
        }
    };
}

impl_int!(I32);
impl_int!(I64);
