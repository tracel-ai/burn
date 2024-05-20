use crate::dialect::{Elem, IntKind, Variable, Vectorization};
use crate::language::{CubeContext, CubeType, ExpandElement, Numeric, PrimitiveVariable};

/// Signed integer. Used as input in int kernels
pub trait Int: Numeric + std::ops::Rem<Output = Self> {
    fn new(val: i64) -> Self;
    fn new_expand(context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType;
}

macro_rules! impl_int {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: <Self as PrimitiveVariable>::Primitive,
            pub vectorization: u8,
        }

        impl CubeType for $type {
            type ExpandType = ExpandElement;
        }

        impl PrimitiveVariable for $type {
            type Primitive = i64;

            fn as_elem() -> Elem {
                Elem::Int(IntKind::$type)
            }

            fn vectorization(&self) -> Vectorization {
                self.vectorization.into()
            }

            fn to_f64(&self) -> f64 {
                self.val as f64
            }

            fn from_f64(val: f64) -> Self {
                Self::new(val as i64)
            }

            fn from_i64(val: i64) -> Self {
                Self::new(val)
            }

            fn from_i64_vec(vec: &[i64]) -> Self {
                Self {
                    // We take only one value, because type implements copy and we can't copy an unknown sized vec
                    // For debugging prefer unvectorized types
                    val: *vec.first().expect("Should be at least one value")
                        as <Self as PrimitiveVariable>::Primitive,
                    vectorization: vec.len() as u8,
                }
            }
        }

        impl Numeric for $type {}

        impl Int for $type {
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
                let new_var = Variable::ConstantScalar(val as f64, Self::as_elem());
                ExpandElement::Plain(new_var)
            }
        }
    };
}

impl_int!(I32);
impl_int!(I64);
