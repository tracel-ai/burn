use burn_jit::gpu::Elem;

use crate::{CubeContext, CubeType, ExpandElement, RuntimeType};

#[derive(Clone, Copy)]
pub struct UInt {
    pub val: <Self as RuntimeType>::Primitive,
    pub vectorization: u8,
}

impl UInt {
    // Use with integer literal
    pub fn new(val: i64) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }
    pub fn new_expand(_context: &mut CubeContext, val: i64) -> <UInt as CubeType>::ExpandType {
        (val as u32).into()
    }
}

impl CubeType for UInt {
    type ExpandType = ExpandElement;
}

impl RuntimeType for UInt {
    type Primitive = i64;
    fn val(&self) -> Self::Primitive {
        self.val
    }
    fn into_elem() -> Elem {
        Elem::UInt
    }
}

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::new(value as <Self as RuntimeType>::Primitive)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::new(value as <Self as RuntimeType>::Primitive)
    }
}
