use crate::dialect::{Elem, Variable, Vectorization};
use crate::language::{CubeContext, CubeType, ExpandElement, Numeric, PrimitiveVariable};
use crate::unexpanded;

#[derive(Clone, Copy)]
/// An unsigned int.
/// Preferred for indexing operations
pub struct UInt {
    pub val: <Self as PrimitiveVariable>::Primitive,
    pub vectorization: u8,
}

impl CubeType for UInt {
    type ExpandType = ExpandElement;
}

impl PrimitiveVariable for UInt {
    type Primitive = u32;

    fn as_elem() -> Elem {
        Elem::UInt
    }

    fn vectorization(&self) -> Vectorization {
        self.vectorization
    }

    fn to_f64(&self) -> f64 {
        self.val as f64
    }

    fn from_f64(val: f64) -> Self {
        Self::new(val as <Self as PrimitiveVariable>::Primitive)
    }

    fn from_i64(val: i64) -> Self {
        Self::new(val as <Self as PrimitiveVariable>::Primitive)
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

impl Numeric for UInt {}

impl UInt {
    pub fn new(_val: <Self as PrimitiveVariable>::Primitive) -> Self {
        unexpanded!()
    }

    pub fn new_expand(
        _context: &mut CubeContext,
        val: <Self as PrimitiveVariable>::Primitive,
    ) -> <Self as CubeType>::ExpandType {
        let new_var = Variable::ConstantScalar(val as f64, Self::as_elem());
        ExpandElement::Plain(new_var)
    }
}

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::new(value as <Self as PrimitiveVariable>::Primitive)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::new(value as <Self as PrimitiveVariable>::Primitive)
    }
}

impl From<i32> for UInt {
    fn from(value: i32) -> Self {
        UInt::new(value as <Self as PrimitiveVariable>::Primitive)
    }
}
