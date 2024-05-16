use burn_jit::gpu::Elem;

use crate::{CubeContext, CubeType, ExpandElement, Numeric, PrimitiveVariable};

#[derive(Clone, Copy)]
/// An unsigned int.
/// Preferred for indexing operations
pub struct UInt {
    pub val: <Self as PrimitiveVariable>::Primitive,
    pub vectorization: u8,
}

impl UInt {
    pub fn from_primitive(val: i64) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }

    pub fn from_primitive_expand(
        _context: &mut CubeContext,
        val: i64,
    ) -> <Self as CubeType>::ExpandType {
        (val as u32).into()
    }
}

impl CubeType for UInt {
    type ExpandType = ExpandElement;
}

impl PrimitiveVariable for UInt {
    type Primitive = i64;
    fn val(&self) -> Self::Primitive {
        self.val
    }
    fn into_elem() -> Elem {
        Elem::UInt
    }
}

impl Numeric for UInt {
    fn lit(val: i64) -> Self {
        Self::from_primitive(val)
    }

    fn lit_expand(context: &mut CubeContext, val: i64) -> <Self as CubeType>::ExpandType {
        Self::from_primitive_expand(context, val)
    }
}

impl From<u32> for UInt {
    fn from(value: u32) -> Self {
        UInt::from_primitive(value as <Self as PrimitiveVariable>::Primitive)
    }
}

impl From<usize> for UInt {
    fn from(value: usize) -> Self {
        UInt::from_primitive(value as <Self as PrimitiveVariable>::Primitive)
    }
}
