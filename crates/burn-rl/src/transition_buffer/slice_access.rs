use burn_core::prelude::*;

/// Trait for types that support tensor-like slice operations,
/// enabling storage in a [`TransitionBuffer`](super::TransitionBuffer).
///
/// Implement this trait for any type that wraps tensors and can be stored
/// in a replay buffer. The buffer uses these operations for:
/// - Pre-allocating storage (`zeros_like`)
/// - Writing transitions (`slice_assign_inplace`)
/// - Sampling batches (`select`)
pub trait SliceAccess<B: Backend>: Clone + Sized {
    /// Create zeroed storage matching the shape of `sample` but with `capacity` rows
    /// along the first dimension.
    fn zeros_like(sample: &Self, capacity: usize, device: &B::Device) -> Self;

    /// Select rows at the given indices along the specified dimension.
    fn select(self, dim: usize, indices: Tensor<B, 1, Int>) -> Self;

    /// Assign `value` at row `index` along the first dimension, in place.
    fn slice_assign_inplace(&mut self, index: usize, value: Self);
}

impl<B: Backend> SliceAccess<B> for Tensor<B, 2> {
    fn zeros_like(sample: &Self, capacity: usize, device: &B::Device) -> Self {
        let feature_dim = sample.dims()[1];
        Tensor::zeros([capacity, feature_dim], device)
    }

    fn select(self, dim: usize, indices: Tensor<B, 1, Int>) -> Self {
        Tensor::select(self, dim, indices)
    }

    fn slice_assign_inplace(&mut self, index: usize, value: Self) {
        self.inplace(|t| t.slice_assign(index..index + 1, value));
    }
}
