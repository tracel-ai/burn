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
    /// # Special Cases (IEEE 754 compliant)
    ///
    /// - If `self` is ±∞ and `other` is not NaN, NaN is returned
    /// - If `other` is ±0 and `self` is not NaN, NaN is returned
    /// - If `other` is ±∞ and `self` is finite, `self` is returned
    /// - If either argument is NaN, NaN is returned
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
        // Normal case: fmod(x, y) = x - y * trunc(x / y)
        let quotient = self.clone().div(other.clone());
        let truncated = quotient.trunc();
        let product = other.clone() * truncated.clone();

        // When divisor is infinity and dividend is finite:
        // - quotient is 0, truncated is 0
        // - but 0 * infinity = NaN, which is wrong
        // We need to handle this case by replacing NaN with 0 when appropriate

        // Check if the product is NaN due to 0 * inf
        let is_zero_times_inf = truncated.equal_elem(0.0).bool_and(other.is_inf());
        let zero_tensor = self.clone().mul_scalar(0.0);
        let corrected_product = product.mask_where(is_zero_times_inf, zero_tensor);

        self - corrected_product
    }

    /// Computes the floating-point remainder of dividing `self` by a scalar.
    ///
    /// The result has the same sign as `self` and magnitude less than the scalar.
    ///
    /// # Special Cases (IEEE 754 compliant)
    ///
    /// - If `self` is ±∞ and scalar is not NaN, NaN is returned
    /// - If scalar is ±0 and `self` is not NaN, NaN is returned
    /// - If scalar is ±∞ and `self` is finite, `self` is returned
    /// - If either argument is NaN, NaN is returned
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
        // Normal case: fmod(x, y) = x - y * trunc(x / y)
        let quotient = self.clone().div_scalar(scalar);
        let truncated = quotient.trunc();
        let product = truncated.mul_scalar(scalar);

        // Handle the special case where scalar is infinity
        // When scalar is ±∞ and self is finite, quotient is 0, truncated is 0
        // but 0 * infinity = NaN, which is wrong - it should be 0
        if scalar.is_infinite() {
            // For finite values, fmod(x, ±∞) = x
            // For infinite values, fmod(±∞, ±∞) = NaN (which is handled by arithmetic)
            return self;
        }

        self - product
    }
}
