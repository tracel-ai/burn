use crate::bridge::Numeric;

/// Trait that lists some floating-point mathematical operations are common to all float-like dtypes.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by the [`Tensor`](crate::Tensor) struct.
pub(crate) trait FloatMathOps: Numeric {
    /// Applies element wise square operation
    ///
    #[cfg_attr(doc, doc = "$y_i = x^{2}$")]
    #[cfg_attr(not(doc), doc = "`y = x^2`")]
    fn square(tensor: Self::Primitive) -> Self::Primitive;

    /// Applies element wise exponential operation.
    ///
    #[cfg_attr(doc, doc = "$y_i = e^{x_i}$")]
    #[cfg_attr(not(doc), doc = "`y = e^x`")]
    fn exp(tensor: Self::Primitive) -> Self::Primitive;

    /// Applies the natural logarithm of one plus the input tensor, element-wise.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \log_e\(x_i + 1\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = log(x_i + 1)`")]
    fn log1p(tensor: Self::Primitive) -> Self::Primitive;

    /// Applies element wise natural log operation *ln*.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \log_e\(x_i\)$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = log(x_i)`")]
    fn log(tensor: Self::Primitive) -> Self::Primitive;

    /// Applies element wise root square operation.
    ///
    #[cfg_attr(doc, doc = r#"$y_i = \sqrt{x_i}$"#)]
    #[cfg_attr(not(doc), doc = "`y_i = sqrt(x_i)`")]
    fn sqrt(tensor: Self::Primitive) -> Self::Primitive;
    /// Returns a new tensor with cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the cosine of a tensor, users should prefer the [`Tensor::cos`](crate::Tensor::cos)
    /// function, which is more high-level and designed for public use.
    fn cos(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the sine of a tensor, users should prefer the [`Tensor::sin`](crate::Tensor::sin)
    /// function, which is more high-level and designed for public use.
    fn sin(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the tangent of a tensor, users should prefer the [`Tensor::tan`](crate::Tensor::tan)
    /// function, which is more high-level and designed for public use.
    fn tan(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with hyperbolic cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the hyperbolic cosine of a tensor, users should prefer the [`Tensor::cosh`](crate::Tensor::cosh)
    /// function, which is more high-level and designed for public use.
    fn cosh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with hyperbolic sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the hyperbolic sine of a tensor, users should prefer the [`Tensor::sinh`](crate::Tensor::sinh)
    /// function, which is more high-level and designed for public use.
    fn sinh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with hyperbolic tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the hyperbolic tangent of a tensor, users should prefer the [`Tensor::tanh`](crate::Tensor::tanh)
    /// function, which is more high-level and designed for public use.
    fn tanh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse cosine of a tensor, users should prefer the [`Tensor::acos`](crate::Tensor::acos)
    /// function, which is more high-level and designed for public use.
    fn acos(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse hyperbolic cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse hyperbolic cosine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse hyperbolic cosine of a tensor, users should prefer the [`Tensor::acosh`](crate::Tensor::acosh)
    /// function, which is more high-level and designed for public use.
    fn acosh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse sine of a tensor, users should prefer the [`Tensor::asin`](crate::Tensor::asin)
    /// function, which is more high-level and designed for public use.
    fn asin(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse hyperbolic sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse hyperbolic sine values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse hyperbolic sine of a tensor, users should prefer the [`Tensor::asinh`](crate::Tensor::asinh)
    /// function, which is more high-level and designed for public use.
    fn asinh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse tangent of a tensor, users should prefer the [`Tensor::atan`](crate::Tensor::atan)
    /// function, which is more high-level and designed for public use.
    fn atan(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a new tensor with inverse hyperbolic tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with inverse hyperbolic tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the inverse hyperbolic tangent of a tensor, users should prefer the [`Tensor::atanh`](crate::Tensor::atanh)
    /// function, which is more high-level and designed for public use.
    fn atanh(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns a tensor with the four-quadrant inverse tangent values of `y` and `x`.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The tensor with y coordinates.
    /// * `rhs` - The tensor with x coordinates.
    ///
    /// # Returns
    ///
    /// A tensor with the four-quadrant inverse tangent values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For the four-quadrant inverse tangent of two tensors, users should prefer the [`Tensor::atan2`](crate::Tensor::atan2)
    /// function, which is more high-level and designed for public use.
    fn atan2(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;
}
