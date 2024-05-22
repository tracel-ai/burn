use std::marker::PhantomData;

use crate::language::{CubeType, ExpandElement};

#[derive(new, Clone, Copy)]
pub struct Tensor<E> {
    _val: PhantomData<E>,
}

impl<C: CubeType> CubeType for Tensor<C> {
    type ExpandType = ExpandElement;
}
