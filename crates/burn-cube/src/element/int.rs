use crate::{CubeContext, CubeType, ExpandElement, Numeric, PrimitiveVariable};
use burn_jit::gpu::{Elem, IntKind, Variable};
use std::rc::Rc;

/// Signed integer. Used as input in int kernels
pub trait Int: Numeric + std::ops::Rem<Output = Self> {
    fn from_primitive(val: i64) -> Self;
    fn from_primitive_expand(context: &mut CubeContext, val: i64)
        -> <Self as CubeType>::ExpandType;
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

        impl PrimitiveVariable for $type {
            type Primitive = i64;
            fn val(&self) -> Self::Primitive {
                self.val
            }
            fn into_elem() -> Elem {
                Elem::Int(IntKind::$type)
            }
        }

        impl Int for $type {
            fn from_primitive(val: i64) -> Self {
                Self {
                    val,
                    vectorization: 1,
                }
            }

            fn from_primitive_expand(
                _context: &mut CubeContext,
                val: i64,
            ) -> <Self as CubeType>::ExpandType {
                let new_var = Variable::ConstantScalar(val as f64, Self::into_elem());
                ExpandElement::new(Rc::new(new_var))
            }
        }

        impl Numeric for $type {
            fn lit(val: i64) -> Self {
                Self::from_primitive(val)
            }

            fn lit_expand(context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType {
                <Self as Int>::from_primitive_expand(context, val)
            }
        }
    };
}

impl_int!(I32);
impl_int!(I64);
