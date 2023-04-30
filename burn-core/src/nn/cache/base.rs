use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

pub(crate) enum CacheState<T> {
    Value(T),
    Empty,
}

pub struct TensorCache<B: Backend, const D: usize> {
    pub(crate) state: CacheState<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> TensorCache<B, D> {
    pub fn empty() -> Self {
        Self {
            state: CacheState::Empty,
        }
    }
}
