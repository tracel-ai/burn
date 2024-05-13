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
    /// Create a Bool from primitive bool
    pub fn new(val: bool) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }

    /// Expand version of new
    pub fn new_expand(_context: &mut CubeContext, val: bool) -> <Bool as CubeType>::ExpandType {
        val.into()
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
