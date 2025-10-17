use crate::{Float, Tensor, TensorPrimitive, backend::Backend};

impl<B, const D: usize> Tensor<B, D, Float>
where
    B: Backend,
{
    /// Truncates the tensor element-wise, rounding toward zero.
    ///
    /// This function returns a new tensor with the same shape as the input tensor,
    /// where each element is truncated toward zero. For positive values, this is
    /// equivalent to floor, and for negative values, it's equivalent to ceil.
    ///
    /// # Special Cases (IEEE 754 compliant)
    ///
    /// - `trunc(±0)` returns ±0 (preserves sign of zero)
    /// - `trunc(±∞)` returns ±∞
    /// - `trunc(NaN)` returns NaN
    ///
    /// # Returns
    ///
    /// A tensor with the same shape where each element has been truncated toward zero.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 1>::from_data([2.3, -1.7, 0.5, -0.5, 3.9], &device);
    ///     let truncated = tensor.trunc();
    ///
    ///     // Result: [2.0, -1.0, 0.0, -0.0, 3.0]
    /// }
    /// ```
    pub fn trunc(self) -> Self {
        Self::new(TensorPrimitive::Float(B::float_trunc(
            self.primitive.tensor(),
        )))
    }
}
