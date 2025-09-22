use crate::{Float, Tensor, backend::Backend};

impl<B, const D: usize> Tensor<B, D, Float>
where
    B: Backend,
{
    /// Computes the floating-point remainder of dividing `self` by `other`.
    ///
    /// The result has the same sign as `self` and magnitude less than `other`.
    /// This is equivalent to the IEEE 754 remainder operation.
    ///
    /// # Arguments
    ///
    /// * `other` - The divisor tensor. Must have the same shape as `self`.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape where each element is the floating-point remainder.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let dividend = Tensor::<B, 1>::from_data([5.3, -5.3, 5.3, -5.3], &device);
    ///     let divisor = Tensor::<B, 1>::from_data([2.0, 2.0, -2.0, -2.0], &device);
    ///     let result = dividend.fmod(divisor);
    ///
    ///     // Result: [1.3, -1.3, 1.3, -1.3]
    /// }
    /// ```
    pub fn fmod(self, other: Self) -> Self {
        // fmod(x, y) = x - y * trunc(x / y)
        // This gives the remainder with the same sign as x
        let quotient = self.clone().div(other.clone());
        let truncated = quotient.trunc();
        self - other * truncated
    }

    /// Computes the floating-point remainder of dividing `self` by a scalar.
    ///
    /// The result has the same sign as `self` and magnitude less than the scalar.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar divisor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape where each element is the floating-point remainder.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 1>::from_data([5.3, -5.3, 7.5, -7.5], &device);
    ///     let result = tensor.fmod_scalar(2.0);
    ///
    ///     // Result: [1.3, -1.3, 1.5, -1.5]
    /// }
    /// ```
    pub fn fmod_scalar(self, scalar: f32) -> Self {
        // fmod(x, y) = x - y * trunc(x / y)
        let quotient = self.clone().div_scalar(scalar);
        let truncated = quotient.trunc();
        self - truncated.mul_scalar(scalar)
    }
}
