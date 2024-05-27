use std::marker::PhantomData;

use crate::{
    dialect::Item,
    language::{CubeType, ExpandElement},
    Comptime, CubeContext, CubeElem,
};

#[derive(Clone, Copy)]
pub struct SharedMemory<T> {
    _val: PhantomData<T>,
}

impl<T: CubeType> CubeType for SharedMemory<T> {
    type ExpandType = ExpandElement;
}

impl<T: CubeElem> SharedMemory<T> {
    pub fn new(_size: Comptime<u32>) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn new_expand(context: &mut CubeContext, size: u32) -> <Self as CubeType>::ExpandType {
        context.create_shared(Item::new(T::as_elem()), size)
    }
}
