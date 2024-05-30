use std::marker::PhantomData;

use crate::language::{CubeType, ExpandElement};

#[derive(new, Clone, Copy)]
pub struct Array<E> {
    _val: PhantomData<E>,
}

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = ExpandElement;
}
