use std::marker::PhantomData;

use crate::{
    language::{CubeType, ExpandElement},
    CubeArg, Runtime, TensorHandle,
};

#[derive(new, Clone, Copy)]
pub struct Tensor<E> {
    _val: PhantomData<E>,
}

impl<C: CubeType> CubeType for Tensor<C> {
    type ExpandType = ExpandElement;
}

impl<C: CubeType> CubeArg for Tensor<C> {
    type ArgType<'a, R: Runtime> = TensorHandle<'a, R>;
}
