use burn_std::Shape;

use crate::{
    Backend, Distribution,
    element::{Element, ElementConversion},
    tensor::{BasicOps, IntTensor},
};

/// Trait that list all operations that can be applied on all numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by the
#[cfg_attr(doc, doc = crate::doc_tensor!())]
#[cfg_attr(not(doc), doc = "`Tensor`")]
/// struct.
pub trait Numeric<B: Backend>: BasicOps<B>
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
    /// For adding tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("add"))]
    #[cfg_attr(not(doc), doc = "`Tensor::add`")]
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
    /// For adding a scalar to a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("add_scalar"))]
    #[cfg_attr(not(doc), doc = "`Tensor::add_scalar`")]
    /// function, which is more high-level and designed for public use.
    fn add_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

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
    /// For subtracting tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sub"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sub`")]
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
    /// For subtracting a scalar from a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sub_scalar"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sub_scalar`")]
    /// function, which is more high-level and designed for public use.
    fn sub_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

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
    /// For dividing tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("div"))]
    #[cfg_attr(not(doc), doc = "`Tensor::div`")]
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
    /// For dividing a tensor by a scalar, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("div_scalar"))]
    #[cfg_attr(not(doc), doc = "`Tensor::div_scalar`")]
    /// function, which is more high-level and designed for public use.
    fn div_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

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
    /// For performing the modulo operation, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("remainder"))]
    #[cfg_attr(not(doc), doc = "`Tensor::remainder`")]
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
    /// For performing the modulo operation, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("remainder_scalar"))]
    #[cfg_attr(not(doc), doc = "`Tensor::remainder_scalar`")]
    /// function, which is more high-level and designed for public use.
    fn remainder_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

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
    /// For multiplying tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("mul"))]
    #[cfg_attr(not(doc), doc = "`Tensor::mul`")]
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
    /// For multiplying a tensor by a scalar, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("mul_scalar"))]
    #[cfg_attr(not(doc), doc = "`Tensor::mul_scalar`")]
    /// function, which is more high-level and designed for public use.
    fn mul_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

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
    /// For negating a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("neg"))]
    #[cfg_attr(not(doc), doc = "`Tensor::neg`")]
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
    /// For getting the signs of the elements of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sign"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sign`")]
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
    /// For summing all the elements of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sum"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sum`")]
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
    /// For summing all the elements of a tensor along a dimension, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sum_dim"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sum_dim`")]
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
    #[cfg_attr(doc, doc = crate::doc_tensor!("prod"))]
    #[cfg_attr(not(doc), doc = "`Tensor::prod`")]
    /// function, which is more high-level and designed for public use.
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
    #[cfg_attr(doc, doc = crate::doc_tensor!("prod_dim"))]
    #[cfg_attr(not(doc), doc = "`Tensor::prod_dim`")]
    /// function, which is more high-level and designed for public use.
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
    /// For computing the mean of all the elements of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("mean"))]
    #[cfg_attr(not(doc), doc = "`Tensor::mean`")]
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
    #[cfg_attr(doc, doc = crate::doc_tensor!("mean_dim"))]
    #[cfg_attr(not(doc), doc = "`Tensor::mean_dim`")]
    /// function, which is more high-level and designed for public use.
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
    #[cfg_attr(doc, doc = crate::doc_tensor!("cumsum"))]
    #[cfg_attr(not(doc), doc = "`Tensor::cumsum`")]
    /// function, which is more high-level and designed for public use.
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
    #[cfg_attr(doc, doc = crate::doc_tensor!("cumprod"))]
    #[cfg_attr(not(doc), doc = "`Tensor::cumprod`")]
    /// function, which is more high-level and designed for public use.
    fn cumprod(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Computes the cumulative minimum of elements along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the cumulative minimum of.
    /// * `dim` - The dimension along which to compute the cumulative minimum.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the minimum
    /// of all elements up to and including that position along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the cumulative minimum of elements along a dimension, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("cummin"))]
    #[cfg_attr(not(doc), doc = "`Tensor::cummin`")]
    /// function, which is more high-level and designed for public use.
    fn cummin(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Computes the cumulative maximum of elements along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the cumulative maximum of.
    /// * `dim` - The dimension along which to compute the cumulative maximum.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the maximum
    /// of all elements up to and including that position along the specified dimension.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For computing the cumulative maximum of elements along a dimension, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("cummax"))]
    #[cfg_attr(not(doc), doc = "`Tensor::cummax`")]
    /// function, which is more high-level and designed for public use.
    fn cummax(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Element-wise greater than comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than the corresponding element
    /// of the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than comparison between two tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("greater"))]
    #[cfg_attr(not(doc), doc = "`Tensor::greater`")]
    /// function, which is more high-level and designed for public use.
    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise greater than comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than the right hand side
    /// scalar, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than comparison between a tensor and a scalar, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("greater_elem"))]
    #[cfg_attr(not(doc), doc = "`Tensor::greater_elem`")]
    /// function, which is more high-level and designed for public use.
    fn greater_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Element-wise greater than or equal comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than or equal to the
    /// corresponding element of the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than or equal comparison between two tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("greater_equal"))]
    #[cfg_attr(not(doc), doc = "`Tensor::greater_equal`")]
    /// function, which is more high-level and designed for public use.
    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise greater than or equal comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is greater than or equal to the right
    /// hand side scalar, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise greater than or equal comparison between a tensor and a scalar, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("greater_equal_elem"))]
    #[cfg_attr(not(doc), doc = "`Tensor::greater_equal_elem`")]
    /// function, which is more high-level and designed for public use.
    fn greater_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Element-wise less than comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is less than the corresponding element of
    /// the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than comparison between two tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("lower"))]
    #[cfg_attr(not(doc), doc = "`Tensor::lower`")]
    /// function, which is more high-level and designed for public use.
    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise less than comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is less than the right hand side scalar,
    /// and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than comparison between a tensor and a scalar, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("lower_elem"))]
    #[cfg_attr(not(doc), doc = "`Tensor::lower_elem`")]
    /// function, which is more high-level and designed for public use.
    fn lower_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Element-wise less than or equal comparison between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding element of the left hand side tensor is less than or equal to the corresponding
    /// element of the right hand side tensor, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than or equal comparison between two tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("lower_equal"))]
    #[cfg_attr(not(doc), doc = "`Tensor::lower_equal`")]
    /// function, which is more high-level and designed for public use.
    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise less than or equal comparison between a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensor, where each element is true if the
    /// corresponding element of the left hand side tensor is less than or equal to the right hand
    /// side scalar, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise less than or equal comparison between a tensor and a scalar, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("lower_equal_elem"))]
    #[cfg_attr(not(doc), doc = "`Tensor::lower_equal_elem`")]
    /// function, which is more high-level and designed for public use.
    fn lower_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive;

    /// Gets the indices of the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the indices of the maximum elements.
    /// * `tensor` - The tensor to get the indices of the maximum elements from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the index of the
    /// maximum element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the indices of the maximum elements of a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("argmax"))]
    #[cfg_attr(not(doc), doc = "`Tensor::argmax`")]
    /// function, which is more high-level and designed for public use.
    fn argmax(tensor: Self::Primitive, dim: usize) -> IntTensor<B>;

    /// Gets the indices of the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the indices of the minimum elements.
    /// * `tensor` - The tensor to get the indices of the minimum elements from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is the index of the
    /// minimum element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the indices of the minimum elements of a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("argmin"))]
    #[cfg_attr(not(doc), doc = "`Tensor::argmin`")]
    /// function, which is more high-level and designed for public use.
    fn argmin(tensor: Self::Primitive, dim: usize) -> IntTensor<B>;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A single-element tensor containing the maximum element of the input tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("max"))]
    #[cfg_attr(not(doc), doc = "`Tensor::max`")]
    /// function, which is more high-level and designed for public use.
    fn max(tensor: Self::Primitive) -> Self::Primitive;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements from.
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the same rank as the input tensor, but the given dim set to a shape of 1.
    /// Each element is the maximum element of the corresponding input dim.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("max_dim"))]
    #[cfg_attr(not(doc), doc = "`Tensor::max_dim`")]
    /// function, which is more high-level and designed for public use.
    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements from.
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tuple containing the maximum element of the input tensor, and a tensor with the same shape
    /// as the input tensor, where each element is the index of the maximum element of the input tensor
    /// at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("max_dim_with_indices"))]
    #[cfg_attr(not(doc), doc = "`Tensor::max_dim_with_indices`")]
    /// function, which is more high-level and designed for public use.
    fn max_dim_with_indices(tensor: Self::Primitive, dim: usize)
    -> (Self::Primitive, IntTensor<B>);

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A single-element tensor containing the maximum absolute element of the input tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum absolute elements of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("max_abs"))]
    #[cfg_attr(not(doc), doc = "`Tensor::max_abs`")]
    /// function, which is more high-level and designed for public use.
    fn max_abs(tensor: Self::Primitive) -> Self::Primitive;

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements from.
    /// * `dim` - The axis along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the same rank as the input tensor, but the given dim set to a shape of 1.
    /// Each element is the maximum absolute element of the corresponding input dim.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the maximum elements of a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("max_abs_dim"))]
    #[cfg_attr(not(doc), doc = "`Tensor::max_abs_dim`")]
    /// function, which is more high-level and designed for public use.
    fn max_abs_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Gets the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements from.
    ///
    /// # Returns
    ///
    /// A single-element tensor containing the minimum element of the input tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the minimum elements of a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("min"))]
    #[cfg_attr(not(doc), doc = "`Tensor::min`")]
    /// function, which is more high-level and designed for public use.
    fn min(tensor: Self::Primitive) -> Self::Primitive;

    /// Gets the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements from.
    /// * `dim` - The axis along which to get the minimum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the same rank as the input tensor, but the given dim set to a shape of 1.
    /// Each element is the minimum element of the corresponding input dim.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the minimum elements of a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("min_dim"))]
    #[cfg_attr(not(doc), doc = "`Tensor::min_dim`")]
    /// function, which is more high-level and designed for public use.
    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive;

    /// Gets the minimum elements and indices of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor and corresponding indices, where
    /// each element is the minimum element of the input tensor at the corresponding index
    /// along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the minimum elements of a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("min_dim_with_indices"))]
    #[cfg_attr(not(doc), doc = "`Tensor::min_dim_with_indices`")]
    /// function, which is more high-level and designed for public use.
    fn min_dim_with_indices(tensor: Self::Primitive, dim: usize)
    -> (Self::Primitive, IntTensor<B>);

    /// Clamp the tensor between the given min and max values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped between the given min and max values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users.
    ///
    /// For clamping a tensor between the given min and max values, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("clamp"))]
    #[cfg_attr(not(doc), doc = "`Tensor::clamp`")]
    /// function, which is more high-level and designed for public use.
    fn clamp(tensor: Self::Primitive, min: Self::Elem, max: Self::Elem) -> Self::Primitive;

    /// Clamps a tensor under a minimum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `min` - The minimum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped under the given min value.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users.
    ///
    /// For clamping a tensor under a minimum value, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("clamp_min"))]
    #[cfg_attr(not(doc), doc = "`Tensor::clamp_min`")]
    /// function, which is more high-level and designed for public use.
    fn clamp_min(tensor: Self::Primitive, min: Self::Elem) -> Self::Primitive;

    /// Clamps a tensor over a maximum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// A new tensor with the values clamped over the given max value.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users.
    ///
    /// For clamping a tensor over a maximum value, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("clamp_max"))]
    #[cfg_attr(not(doc), doc = "`Tensor::clamp_max`")]
    /// function, which is more high-level and designed for public use.
    fn clamp_max(tensor: Self::Primitive, max: Self::Elem) -> Self::Primitive;

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
    /// For calculating abs of the elements of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("abs"))]
    #[cfg_attr(not(doc), doc = "`Tensor::abs`")]
    /// function, which is more high-level and designed for public use.
    fn abs(tensor: Self::Primitive) -> Self::Primitive;

    /// Element-wise power of a tensor to a float tensor
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Element-wise power of a tensor
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;

    /// Element-wise power of a tensor to a scalar float
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powf_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

    /// Element-wise power of a tensor to a scalar int
    ///
    /// # Arguments
    /// * `tensor` - The tensor to apply power to.
    /// * `power` - The power to apply to the tensor.
    fn powi_scalar<E: ElementConversion>(lhs: Self::Primitive, rhs: E) -> Self::Primitive;

    /// Create a random tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the output tensor.
    /// * `distribution` - The distribution used to sample.
    /// * `device` - The device to use.
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
    /// Users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("random"))]
    #[cfg_attr(not(doc), doc = "`Tensor::random`")]
    /// function, which is more high-level and designed for public use.
    fn random(shape: Shape, distribution: Distribution, device: &B::Device) -> Self::Primitive;

    /// Sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    /// * `descending` - The sorting order.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where the elements are sorted by value.
    ///
    /// # Remarks
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sort"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sort`")]
    /// function, which is more high-level and designed for public use.
    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive;

    /// Sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    /// * `descending` - The sorting order.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor and corresponding indices, where
    /// the elements are sorted by value and the indices map back to the original input tensor.
    ///
    /// # Remarks
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For sorting the elements of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("sort_with_indices"))]
    #[cfg_attr(not(doc), doc = "`Tensor::sort_with_indices`")]
    /// function, which is more high-level and designed for public use.
    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, IntTensor<B>);

    /// Returns the indices that sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    /// * `descending` - The sorting order.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor the indices map back to the original input tensor.
    ///
    /// # Remarks
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("argsort"))]
    #[cfg_attr(not(doc), doc = "`Tensor::argsort`")]
    /// function, which is more high-level and designed for public use.
    fn argsort(tensor: Self::Primitive, dim: usize, descending: bool) -> IntTensor<B>;

    /// Applies the matrix multiplication operation.
    ///
    /// ```math
    /// C = AB
    /// ```
    fn matmul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive;
}
