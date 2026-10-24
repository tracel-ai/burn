use burn_backend::Scalar;
use burn_backend::ops::FloatTensorOps;
use burn_dispatch::Dispatch;

use crate::check;
use crate::check::TensorCheck;
use crate::{
    Float, Tensor,
    ops::{BridgeTensor, Resolved},
};

impl<const D: usize> Tensor<D, Float> {
    /// Computes the floating-point remainder of dividing `self` by `other`.
    ///
    /// The result has the same sign as `self` and magnitude less than `|other|`.
    /// This matches C's `fmod`: the quotient is truncated toward zero, whereas `remainder`
    /// floors it and takes the sign of `other`.
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
    /// use burn_tensor::Tensor;
    ///
    /// let device = Default::default();
    /// let dividend = Tensor::<1>::from_data([5.3, -5.3, 5.3, -5.3], &device);
    /// let divisor = Tensor::<1>::from_data([2.0, 2.0, -2.0, -2.0], &device);
    /// let result = dividend.fmod(divisor);
    ///
    /// // Result: [1.3, -1.3, 1.3, -1.3]
    /// ```
    pub fn fmod(self, other: Self) -> Self {
        check!(TensorCheck::binary_ops_ew("Fmod", &self, &other));
        Self::new(fmod_impl(self.primitive, other.primitive))
    }

    /// Computes the floating-point remainder of dividing `self` by a scalar.
    ///
    /// The result has the same sign as `self` and magnitude less than `|scalar|`.
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
    /// use burn_tensor::Tensor;
    ///
    /// let device = Default::default();
    /// let tensor = Tensor::<1>::from_data([5.3, -5.3, 7.5, -7.5], &device);
    /// let result = tensor.fmod_scalar(2.0);
    ///
    /// // Result: [1.3, -1.3, 1.5, -1.5]
    /// ```
    pub fn fmod_scalar(self, scalar: f32) -> Self {
        let scalar = Scalar::new(scalar, &self.dtype());
        Self::new(fmod_scalar_impl(self.primitive, scalar))
    }
}

fn fmod_impl(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
    let resolved = Resolved::combine([&lhs, &rhs]);
    resolved.output(Dispatch::float_fmod(
        resolved.operand(lhs),
        resolved.operand(rhs),
    ))
}

fn fmod_scalar_impl(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
    let resolved = Resolved::combine([&lhs]);
    resolved.output(Dispatch::float_fmod_scalar(resolved.operand(lhs), rhs))
}
