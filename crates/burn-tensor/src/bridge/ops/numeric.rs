use burn_std::{DType, Shape};

use crate::{BackendTypes, Distribution, Scalar, bridge::BasicOps, element::Element};

/// Trait that list all operations that can be applied on all numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by the [`Tensor`](crate::Tensor) struct.
pub trait Numeric<B: BackendTypes>: BasicOps<B>
where
    Self::Elem: Element,
{
    /// Adds two tensors together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The sum of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For adding tensors, users should prefer the [`Tensor::add`](crate::Tensor::add)
    /// function, which is more high-level and designed for public use.
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Adds a scalar to a tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The sum of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For adding a scalar to a tensor, users should prefer the [`Tensor::add_scalar`](crate::Tensor::add_scalar)
    /// function, which is more high-level and designed for public use.
    fn add_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive;

    /// Subtracts two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The difference of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For subtracting tensors, users should prefer the [`Tensor::sub`](crate::Tensor::sub)
    /// function, which is more high-level and designed for public use.
    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Subtracts a scalar from a tensor element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The difference of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For subtracting a scalar from a tensor, users should prefer the [`Tensor::sub_scalar`](crate::Tensor::sub_scalar)
    /// function, which is more high-level and designed for public use.
    fn sub_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive;

    /// Divides two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The quotient of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For dividing tensors, users should prefer the [`Tensor::div`](crate::Tensor::div)
    /// function, which is more high-level and designed for public use.
    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Divides a tensor by a scalar element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The quotient of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For dividing a tensor by a scalar, users should prefer the [`Tensor::div_scalar`](crate::Tensor::div_scalar)
    /// function, which is more high-level and designed for public use.
    fn div_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive;

    /// Computes the modulo element-wise. The result is the *signed* remainder of the division and its absolute value is
    /// less than that of the divisor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The dividend.
    /// * `rhs` - The divisor.
    ///
    /// # Returns
    ///
    /// The modulo of the input tensor with the divisor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For performing the modulo operation, users should prefer the [`Tensor::remainder`](crate::Tensor::remainder)
    /// function, which is more high-level and designed for public use.
    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Computes the modulo element-wise. The result is the *signed* remainder of the division and its absolute value is
    /// less than that of the divisor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The dividend.
    /// * `rhs` - The divisor.
    ///
    /// # Returns
    ///
    /// The modulo of the input tensor with the divisor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For performing the modulo operation, users should prefer the [`Tensor::remainder_scalar`](crate::Tensor::remainder_scalar)
    /// function, which is more high-level and designed for public use.
    fn remainder_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive;

    /// Multiplies two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The product of the two tensors.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For multiplying tensors, users should prefer the [`Tensor::mul`](crate::Tensor::mul)
    /// function, which is more high-level and designed for public use.
    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Multiplies a tensor by a scalar element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The product of the tensor and the scalar.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For multiplying a tensor by a scalar, users should prefer the [`Tensor::mul_scalar`](crate::Tensor::mul_scalar)
    /// function, which is more high-level and designed for public use.
    fn mul_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive;

    /// Negates a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to negate.
    ///
    /// # Returns
    ///
    /// The negated tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For negating a tensor, users should prefer the [`Tensor::neg`](crate::Tensor::neg)
    /// function, which is more high-level and designed for public use.
    fn neg(tensor: Self::Primitive) -> Self::Primitive;

    /// Returns the signs of the elements of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The signs of the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the signs of the elements of a tensor, users should prefer the [`Tensor::sign`](crate::Tensor::sign)
    /// function, which is more high-level and designed for public use.
    fn sign(tensor: Self::Primitive) -> Self::Primitive;

    /// Sums all the elements of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    ///
    /// # Returns
    ///
    /// The sum of all the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For summing all the elements of a tensor, users should prefer the [`Tensor::sum`](crate::Tensor::sum)
    /// function, which is more high-level and designed for public use.
    fn sum(tensor: Self::Primitive) -> Self::Primitive;

    /// Sums all the elements of the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    /// * `dim` - The dimension along which to sum.
    ///
    /// # Returns
    ///
    /// The sum of all the elements of the tensor along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For summing all the elements of a tensor along a dimension, users should prefer the [`Tensor::sum_dim`](crate::Tensor::sum_dim)
    /// function, which is more high-level and designed for public use.
    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Computes the product of all the elements of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the product of.
    ///
    /// # Returns
    ///
    /// The product of all the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the product of all the elements of a tensor, users should prefer the
    /// [`Tensor::prod`](crate::Tensor::prod) function, which is more high-level and designed for public use.
    fn prod(tensor: Self::Primitive) -> Self::Primitive;

    /// Computes the product of all the elements of the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the product of.
    /// * `dim` - The dimension along which to compute the product.
    ///
    /// # Returns
    ///
    /// The product of all the elements of the tensor along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the product of all the elements of a tensor along a dimension, users should prefer the
    /// [`Tensor::prod_dim`](crate::Tensor::prod_dim) function, which is more high-level and designed for public use.
    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Computes the mean of all the elements of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    ///
    /// # Returns
    ///
    /// The mean of all the elements of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the mean of all the elements of a tensor, users should prefer the [`Tensor::mean`](crate::Tensor::mean)
    /// function, which is more high-level and designed for public use.
    fn mean(tensor: Self::Primitive) -> Self::Primitive;

    /// Computes the mean of all the elements of the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    /// * `dim` - The dimension along which to compute the mean.
    ///
    /// # Returns
    ///
    /// The mean of all the elements of the tensor along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the mean of all the elements of a tensor along a dimension, users should prefer the
    /// [`Tensor::mean_dim`](crate::Tensor::mean_dim) function, which is more high-level and designed for public use.
    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Computes the cumulative sum of elements along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the cumulative sum of.
    /// * `dim` - The dimension along which to compute the cumulative sum.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the cumulative sum
    /// of all elements up to and including that position along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the cumulative sum of elements along a dimension, users should prefer the
    /// [`Tensor::cumsum`](crate::Tensor::cumsum) function, which is more high-level and designed for public use.
    fn cumsum(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Computes the cumulative product of elements along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the cumulative product of.
    /// * `dim` - The dimension along which to compute the cumulative product.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the cumulative product
    /// of all elements up to and including that position along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the cumulative product of elements along a dimension, users should prefer the
    /// [`Tensor::cumprod`](crate::Tensor::cumprod) function, which is more high-level and designed for public use.
    fn cumprod(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Calculate absolute value on all elements of a tensor
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to apply abs to.
    ///
    /// # Returns
    ///
    /// A tensor with absolute values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For calculating abs of the elements of a tensor, users should prefer the [`Tensor::abs`](crate::Tensor::abs)
    /// function, which is more high-level and designed for public use.
    fn abs(tensor: Self::Primitive) -> Self::Primitive;

    /// Element-wise power of a tensor
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Element-wise power of a tensor to a scalar int
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powi_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive;

    /// Create a random tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the output tensor.
    /// * `distribution` - The distribution used to sample.
    /// * `device` - The device to use.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// A new tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [`Tensor::random`](crate::Tensor::random)
    /// function, which is more high-level and designed for public use.
    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &B::Device,
        dtype: DType,
    ) -> Self::Primitive;

    /// Applies the matrix multiplication operation.
    ///
    /// ```math
    /// C = AB
    /// ```
    fn matmul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;
}
