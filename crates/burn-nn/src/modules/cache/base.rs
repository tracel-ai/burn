use burn_core as burn;

use burn::tensor::Tensor;

pub(crate) enum CacheState<T> {
    Value(T),
    Empty,
}

/// A cache for a tensor.
pub struct TensorCache<const D: usize> {
    pub(crate) state: CacheState<Tensor<D>>,
}

impl<const D: usize> TensorCache<D> {
    /// Creates a new empty cache.
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
