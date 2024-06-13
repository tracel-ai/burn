use std::marker::PhantomData;

use crate::{
    frontend::{indexation::Index, CubeContext, CubeElem, CubeType},
    ir::Item,
};

use super::{ExpandElement, Init};

#[derive(Clone, Copy)]
pub struct SharedMemory<T: CubeType> {
    _val: PhantomData<T>,
}

#[derive(Clone)]
pub struct SharedMemoryExpand<T: CubeElem> {
    pub val: <T as CubeType>::ExpandType,
}

impl<T: CubeElem> From<SharedMemoryExpand<T>> for ExpandElement {
    fn from(shared_memory_expand: SharedMemoryExpand<T>) -> Self {
        shared_memory_expand.val
    }
}

impl<T: CubeElem> Init for SharedMemoryExpand<T> {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

impl<T: CubeElem> CubeType for SharedMemory<T> {
    type ExpandType = SharedMemoryExpand<T>;
}

impl<T: CubeElem + Clone> SharedMemory<T> {
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
}
