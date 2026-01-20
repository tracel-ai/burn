use crate::{
    Backend, Scalar,
    tensor::{IntTensor, Numeric},
};

/// Trait that list all operations that can be applied on all numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by the
#[cfg_attr(doc, doc = crate::doc_tensor!())]
#[cfg_attr(not(doc), doc = "`Tensor`")]
/// struct.
pub trait Ordered<B: Backend>: Numeric<B> {
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
    fn greater_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive;

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
    fn greater_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive;

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
    fn lower_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive;

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
    fn lower_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive;

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
    fn clamp(tensor: Self::Primitive, min: Scalar, max: Scalar) -> Self::Primitive;

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
    fn clamp_min(tensor: Self::Primitive, min: Scalar) -> Self::Primitive;

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
    fn clamp_max(tensor: Self::Primitive, max: Scalar) -> Self::Primitive;
}
