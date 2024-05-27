use std::marker::PhantomData;

use crate::{
    dialect::Item,
    language::{indexation::Index, CubeType, ExpandElement},
    CubeContext, CubeElem,
};

#[derive(Clone, Copy)]
pub struct SharedMemory<T> {
    _val: PhantomData<T>,
}

impl<T: CubeType> CubeType for SharedMemory<T> {
    type ExpandType = ExpandElement;
}

impl<T: CubeElem> SharedMemory<T> {
    pub fn new<S: Index>(_size: S) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn new_expand<S: Index>(
        context: &mut CubeContext,
        size: S,
    ) -> <Self as CubeType>::ExpandType {
        let size = size.value();
        let size = match size {
            crate::dialect::Variable::ConstantScalar(val, _) => val as u32,
            _ => panic!("Shared memory need constant initialization value"),
        };
        context.create_shared(Item::new(T::as_elem()), size)
    }
}
