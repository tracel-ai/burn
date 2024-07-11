use std::marker::PhantomData;

use crate::{
    frontend::{indexation::Index, CubeContext, CubePrimitive, CubeType},
    ir::Item,
};

use super::{ExpandElementTyped, Init, UInt};

#[derive(Clone, Copy)]
pub struct SharedMemory<T: CubeType> {
    _val: PhantomData<T>,
}

impl<T: CubePrimitive> Init for ExpandElementTyped<SharedMemory<T>> {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl<T: CubePrimitive> CubeType for SharedMemory<T> {
    type ExpandType = ExpandElementTyped<SharedMemory<T>>;
}

impl<T: CubePrimitive + Clone> SharedMemory<T> {
    pub fn new<S: Index>(_size: S) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn vectorized<S: Index>(_size: S, _vectorization_factor: UInt) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn __expand_vectorized<S: Index>(
        context: &mut CubeContext,
        size: S,
        vectorization_factor: UInt,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar { value, .. } => value as u32,
            _ => panic!("Shared memory need constant initialization value"),
        };
        let var = context.create_shared(
            Item::vectorized(T::as_elem(), vectorization_factor.val as u8),
            size,
        );
        ExpandElementTyped::new(var)
    }

    pub fn __expand_new<S: Index>(
        context: &mut CubeContext,
        size: S,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar { value, .. } => value as u32,
            _ => panic!("Shared memory need constant initialization value"),
        };
        let var = context.create_shared(Item::new(T::as_elem()), size);
        ExpandElementTyped::new(var)
    }
}
