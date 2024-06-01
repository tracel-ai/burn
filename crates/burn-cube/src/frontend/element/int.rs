use crate::frontend::{CubeContext, CubeElem, CubeType, ExpandElement, Numeric};
use crate::ir::{Elem, IntKind, Variable};

/// Signed integer. Used as input in int kernels
pub trait Int: Numeric + std::ops::Rem<Output = Self> {
    fn new(val: i64) -> Self;
    fn new_expand(context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType;
}

macro_rules! impl_int {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: i64,
            pub vectorization: u8,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElement;
        }

        impl CubeElem for $type {
            fn as_elem() -> Elem {
                Elem::Int(IntKind::$type)
            }
        }

        impl Numeric for $type {}

        impl Int for $type {
            fn new(val: i64) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }

            fn new_expand(_context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType {
                let new_var = Variable::ConstantScalar(val as f64, Self::as_elem());
                ExpandElement::Plain(new_var)
            }
        }
    };
}

impl_int!(I32);
impl_int!(I64);
