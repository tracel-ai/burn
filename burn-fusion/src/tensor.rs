use burn_tensor::{backend::Backend, Shape};
use core::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct FusionTensor<B: Backend, E> {
    shape: Vec<usize>,
    device: B::Device,
    id: usize,
    _elem: PhantomData<E>,
}

impl<B: Backend, E> FusionTensor<B, E> {
    pub(crate) fn shape<const D: usize>(&self) -> Shape<D> {
        Shape::from(self.shape.clone())
    }
}
