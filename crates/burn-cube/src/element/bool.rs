use burn_jit::gpu::Elem;

use crate::{CubeContext, CubeType, ExpandElement, RuntimeType};

#[derive(Clone, Copy)]
pub struct Bool {
    pub val: bool,
    pub vectorization: u8,
}

impl CubeType for Bool {
    type ExpandType = ExpandElement;
}

impl Bool {
    pub fn new(val: bool) -> Self {
        Self {
            val,
            vectorization: 1,
        }
    }
    pub fn new_expand(_context: &mut CubeContext, val: bool) -> <Bool as CubeType>::ExpandType {
        val.into()
    }
}

impl RuntimeType for Bool {
    type Primitive = bool;
    fn val(&self) -> Self::Primitive {
        self.val
    }

    fn into_elem() -> Elem {
        Elem::Bool
    }
}
