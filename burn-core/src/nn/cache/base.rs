use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

pub(crate) enum CacheState<T> {
    Value(T),
    Empty,
}

/// A cache for a tensor.
pub struct TensorCache<B: Backend, const D: usize> {
    pub(crate) state: CacheState<Tensor<B, D>>,
}

impl<B: Backend, const D: usize> TensorCache<B, D> {
    /// Gets empty cache.
    ///
    /// # Returns
    ///
    /// The empty cache.
    pub fn empty() -> Self {
        Self {
            state: CacheState::Empty,
        }
    }
}
