use crate::dialect::{Elem, Variable};
use crate::language::{CubeContext, CubeElem, CubeType, ExpandElement, Numeric};

#[derive(Clone, Copy)]
/// An unsigned int.
/// Preferred for indexing operations
pub struct UInt {
    pub val: u32,
    pub vectorization: u8,
}

impl CubeType for UInt {
    type ExpandType = ExpandElement;
}

impl CubeElem for UInt {
    fn as_elem() -> Elem {
        Elem::UInt
    }
}

impl Numeric for UInt {}

impl UInt {
    pub fn new(val: u32) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }

    pub fn new_expand(_context: &mut CubeContext, val: u32) -> <Self as CubeType>::ExpandType {
        let new_var = Variable::ConstantScalar(val as f64, Self::as_elem());
        ExpandElement::Plain(new_var)
    }
}

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::new(value)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::new(value as u32)
    }
}

impl From<i32> for UInt {
    fn from(value: i32) -> Self {
        UInt::new(value as u32)
    }
}
