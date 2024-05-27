use std::marker::PhantomData;

use crate::{
    dialect::Item,
    language::{CubeType, ExpandElement},
    ComptimeIndex, CubeContext, CubeElem,
};

#[derive(Clone, Copy)]
pub struct SharedMemory<T> {
    _val: PhantomData<T>,
}

impl<T: CubeType> CubeType for SharedMemory<T> {
    type ExpandType = ExpandElement;
}

impl<T: CubeElem> SharedMemory<T> {
    pub fn new<S: ComptimeIndex>(_size: S) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn new_expand<S: ComptimeIndex>(
        context: &mut CubeContext,
        size: S,
    ) -> <Self as CubeType>::ExpandType {
        context.create_shared(Item::new(T::as_elem()), size.value())
    }
}
