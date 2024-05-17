use std::rc::Rc;

use crate::dialect::{Elem, Variable};
use crate::language::{CubeContext, CubeType, ExpandElement, Numeric, PrimitiveVariable};

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

    fn into_elem() -> Elem {
        Elem::UInt
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
}

impl Numeric for UInt {}

impl UInt {
    pub fn new(val: <Self as PrimitiveVariable>::Primitive) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }
    pub fn new_expand(
        _context: &mut CubeContext,
        val: <Self as PrimitiveVariable>::Primitive,
    ) -> <Self as CubeType>::ExpandType {
        let new_var = Variable::ConstantScalar(val as f64, Self::into_elem());
        ExpandElement::new(Rc::new(new_var))
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
