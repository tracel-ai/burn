use burn_jit::gpu::Elem;

use crate::{CubeContext, CubeType, ExpandElement, PrimitiveVariable};

#[derive(Clone, Copy)]
/// Boolean type for kernels
pub struct Bool {
    pub val: bool,
    pub vectorization: u8,
}

impl CubeType for Bool {
    type ExpandType = ExpandElement;
}

impl Bool {
    /// Make a boolean literal
    pub fn lit(val: bool) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }

    /// Expand version of lit
    pub fn lit_expand(_context: &mut CubeContext, val: bool) -> <Self as CubeType>::ExpandType {
        val.into()
    }

    /// Create a Bool from primitive bool
    pub fn from_primitive(val: bool) -> Self {
        Self::lit(val)
    }

    /// Expand version of from_primitive
    pub fn from_primitive_expand(
        context: &mut CubeContext,
        val: bool,
    ) -> <Self as CubeType>::ExpandType {
        Self::lit_expand(context, val)
    }
}

impl PrimitiveVariable for Bool {
    type Primitive = bool;

    fn val(&self) -> Self::Primitive {
        self.val
    }

    fn into_elem() -> Elem {
        Elem::Bool
    }
}
