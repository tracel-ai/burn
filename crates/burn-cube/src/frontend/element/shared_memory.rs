use std::marker::PhantomData;

use crate::{
    frontend::{indexation::Index, CubeContext, CubePrimitive, CubeType},
    ir::Item,
};

use super::{ExpandElement, Init, UInt};

#[derive(Clone, Copy)]
pub struct SharedMemory<T: CubeType> {
    _val: PhantomData<T>,
}

#[derive(Clone)]
pub struct SharedMemoryExpand<T: CubePrimitive> {
    pub val: <T as CubeType>::ExpandType,
}

impl<T: CubePrimitive> From<SharedMemoryExpand<T>> for ExpandElement {
    fn from(shared_memory_expand: SharedMemoryExpand<T>) -> Self {
        shared_memory_expand.val
    }
}

impl<T: CubePrimitive> Init for SharedMemoryExpand<T> {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl<T: CubePrimitive> CubeType for SharedMemory<T> {
    type ExpandType = SharedMemoryExpand<T>;
}

impl<T: CubePrimitive + Clone> SharedMemory<T> {
    pub fn new<S: Index>(_size: S) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn new_expand<S: Index>(
        context: &mut CubeContext,
        size: S,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar(val, _) => val as u32,
            _ => panic!("Shared memory need constant initialization value"),
        };
        context.create_shared(Item::new(T::as_elem()), size)
    }

    pub fn vectorized<S: Index>(_size: S, _vectorization_factor: UInt) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn vectorized_expand<S: Index>(
        context: &mut CubeContext,
        size: S,
        vectorization_factor: UInt,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::ir::Variable::ConstantScalar(val, _) => val as u32,
            _ => panic!("Shared memory need constant initialization value"),
        };
        context.create_shared(
            Item::vectorized(T::as_elem(), vectorization_factor.val as u8),
            size,
        )
    }
}
