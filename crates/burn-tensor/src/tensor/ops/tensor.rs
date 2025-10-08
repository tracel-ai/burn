use super::cat::cat_with_slice_assign;
use super::grid_sample::float_grid_sample_2d_bilinear;
use super::repeat_dim::repeat_with_slice_assign;
use super::{BoolTensor, Device, FloatElem, FloatTensor, IntElem, IntTensor};
use crate::ops::InterpolateMode;
use crate::{Distribution, ElementConversion, Float, TensorData, backend::Backend, tensor::Shape};
use crate::{FloatDType, TensorMetadata, TensorPrimitive};
use alloc::vec::Vec;

use crate::{argsort, sort, sort_with_indices};

/// Operations on float tensors.
pub trait FloatTensorOps<B: Backend> {
    /// Creates a new tensor from the data structure.
    ///
    /// # Arguments
    ///
    /// * `data` - The data structure.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given data.
    fn float_from_data(data: TensorData, device: &Device<B>) -> FloatTensor<B>;

    /// Creates a new tensor with random values.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `distribution` - The distribution to sample from.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and random values.
    fn float_random(shape: Shape, distribution: Distribution, device: &Device<B>)
    -> FloatTensor<B>;

    /// Creates a new tensor with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and zeros.
    fn float_zeros(shape: Shape, device: &Device<B>, dtype: FloatDType) -> FloatTensor<B> {
        Self::float_from_data(TensorData::full_dtype(shape, 0, dtype.into()), device)
    }

    /// Creates a new tensor with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and ones.
    fn float_ones(shape: Shape, device: &Device<B>, dtype: FloatDType) -> FloatTensor<B> {
        Self::float_from_data(TensorData::full_dtype(shape, 1, dtype.into()), device)
    }

    /// Creates a tensor filled with given value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `fill_value` - The value with which to fill the tensor.
    /// * `device` - The device to create the tensor on.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The tensor filled with given value
    fn float_full(
        shape: Shape,
        fill_value: FloatElem<B>,
        device: &Device<B>,
        dtype: FloatDType,
    ) -> FloatTensor<B> {
        Self::float_from_data(
            TensorData::full_dtype(shape, fill_value, dtype.into()),
            device,
        )
    }

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn float_into_data(tensor: FloatTensor<B>) -> impl Future<Output = TensorData> + Send;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn float_device(tensor: &FloatTensor<B>) -> Device<B>;

    /// Moves the tensor to the given device.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `device` - The device to move the tensor to.
    ///
    /// # Returns
    ///
    /// The tensor on the given device.
    fn float_to_device(tensor: FloatTensor<B>, device: &Device<B>) -> FloatTensor<B>;

    /// Converts float tensor to int tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The int tensor with the same data as the float tensor.
    fn float_into_int(tensor: FloatTensor<B>) -> IntTensor<B>;

    /// Creates an empty tensor with the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The empty tensor with the given shape.
    fn float_empty(shape: Shape, device: &Device<B>, dtype: FloatDType) -> FloatTensor<B>;

    /// Repeat the tensor along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to repeat.
    /// * `times` - The number of times to repeat the dimension.
    ///
    /// # Returns
    ///
    /// The tensor with the given dimension repeated.
    fn float_repeat_dim(tensor: FloatTensor<B>, dim: usize, times: usize) -> FloatTensor<B> {
        repeat_with_slice_assign::<B, Float>(TensorPrimitive::Float(tensor), dim, times).tensor()
    }

    /// Adds two tensors together.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of adding the two tensors together.
    fn float_add(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;

    /// Adds a scalar to a tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of adding the scalar to the tensor.
    fn float_add_scalar(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> FloatTensor<B>;

    /// Clamps a tensor under a minimum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `min` - The minimum value.
    ///
    /// # Returns
    ///
    /// The clamped tensor.
    fn float_clamp_min(tensor: FloatTensor<B>, min: FloatElem<B>) -> FloatTensor<B> {
        // Default implementation
        let mask = Self::float_lower_elem(tensor.clone(), min);
        B::float_mask_fill(tensor, mask, min)
    }

    /// Clamps a tensor over a maximum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// The clamped tensor.
    fn float_clamp_max(tensor: FloatTensor<B>, max: FloatElem<B>) -> FloatTensor<B> {
        // Default implementation
        let mask = Self::float_greater_elem(tensor.clone(), max);
        B::float_mask_fill(tensor, mask, max)
    }

    /// Clamps a tensor between a minimum and maximum value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to clamp.
    /// * `min` - The minimum value.
    /// * `max` - The maximum value.
    ///
    /// # Returns
    ///
    /// The clamped tensor.
    fn float_clamp(tensor: FloatTensor<B>, min: FloatElem<B>, max: FloatElem<B>) -> FloatTensor<B> {
        // Default implementation
        Self::float_clamp_min(Self::float_clamp_max(tensor, max), min)
    }

    /// Subtracts two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of subtracting the two tensors.
    fn float_sub(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;

    /// Subtracts a scalar from a tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of subtracting the scalar from the tensor.
    fn float_sub_scalar(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> FloatTensor<B>;

    /// Multiplies two tensors together element-wise.
    fn float_mul(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;

    /// Multiplies a tensor by a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of multiplying the tensor by the scalar.
    fn float_mul_scalar(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> FloatTensor<B>;

    /// Divides two tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of dividing the two tensors.
    fn float_div(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;

    /// Divides a tensor by a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of dividing the tensor by the scalar.
    fn float_div_scalar(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> FloatTensor<B>;

    /// Computes the remainder of division between two tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The element-wise remainder when dividing `lhs` by `rhs`.
    fn float_remainder(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;

    /// Computes the modulus of a tensor given a scalar.
    ///
    /// # Arguments
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of applying the modulus of the scalar to the tensor.
    fn float_remainder_scalar(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> FloatTensor<B>;

    /// Multiplies two tensors together using matrix multiplication.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of multiplying the two tensors together using matrix multiplication.
    fn float_matmul(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;

    /// Computes the cross product of two tensors along a given dimension.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    /// * `dim` - The dimension to compute the cross product along.
    ///
    /// # Returns
    ///
    /// The cross product of the two tensors.
    fn float_cross(lhs: FloatTensor<B>, rhs: FloatTensor<B>, dim: usize) -> FloatTensor<B>;

    /// Negates a tensor element-wise.
    fn float_neg(tensor: FloatTensor<B>) -> FloatTensor<B> {
        Self::float_mul_scalar(tensor, (-1.0_f32).elem::<FloatElem<B>>())
    }

    /// Calculates the reciprocals element-wise
    fn float_recip(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Transposes a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn float_transpose(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let ndims = tensor.shape().num_dims();
        Self::float_swap_dims(tensor, ndims - 2, ndims - 1)
    }

    /// Swaps two dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to swap the dimensions of.
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions swapped.
    fn float_swap_dims(tensor: FloatTensor<B>, dim1: usize, dim2: usize) -> FloatTensor<B>;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn float_permute(tensor: FloatTensor<B>, axes: &[usize]) -> FloatTensor<B>;

    /// Reverse the order of elements in a tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reverse.
    /// * `axes` - The axes to reverse.
    ///
    /// The tensor with the elements reversed.
    fn float_flip(tensor: FloatTensor<B>, axes: &[usize]) -> FloatTensor<B>;

    /// Reshapes a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reshape.
    /// * `shape` - The new shape of the tensor.
    ///
    /// # Returns
    ///
    /// The tensor with the new shape.
    fn float_reshape(tensor: FloatTensor<B>, shape: Shape) -> FloatTensor<B>;

    /// Gather elements from a tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to gather from.
    /// * `tensor` - The tensor to gather from.
    /// * `indices` - The indices to gather.
    ///
    /// # Returns
    ///
    /// The gathered elements.
    fn float_gather(dim: usize, tensor: FloatTensor<B>, indices: IntTensor<B>) -> FloatTensor<B>;

    /// Scatter elements into a tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to scatter into.
    /// * `tensor` - The tensor to scatter into.
    /// * `indices` - The indices to scatter into.
    /// * `value` - The value to scatter.
    ///
    /// # Returns
    ///
    /// The tensor with the scattered elements.
    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<B>,
        indices: IntTensor<B>,
        value: FloatTensor<B>,
    ) -> FloatTensor<B>;

    /// Select tensor elements along the given dimension corresponding for the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices to select.
    ///
    /// # Returns
    ///
    /// The selected elements.
    fn float_select(tensor: FloatTensor<B>, dim: usize, indices: IntTensor<B>) -> FloatTensor<B>;

    /// Assign the selected elements along the given dimension corresponding for the given indices
    /// to the given value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices to select.
    /// * `value` - The value to assign.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to the given value.
    fn float_select_assign(
        tensor: FloatTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
        value: FloatTensor<B>,
    ) -> FloatTensor<B>;

    /// Select tensor elements corresponding to the given slices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `slices` - The slices specifying ranges and steps for each dimension.
    ///
    /// # Returns
    ///
    /// The selected elements in a new tensor.
    fn float_slice(tensor: FloatTensor<B>, slices: &[crate::Slice]) -> FloatTensor<B>;

    /// Assign the selected elements corresponding to the given slices to the given value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `ranges` - The ranges to select.
    /// * `value` - The value to assign.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to the given value.
    fn float_slice_assign(
        tensor: FloatTensor<B>,
        slices: &[crate::Slice],
        value: FloatTensor<B>,
    ) -> FloatTensor<B>;

    /// Update the given tensor with the value tensor where the mask is true.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `mask` - The boolean mask to select with.
    /// * `value` - The value to assign to the selected elements from the value tensor.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to the given value.
    fn float_mask_where(
        tensor: FloatTensor<B>,
        mask: BoolTensor<B>,
        value: FloatTensor<B>,
    ) -> FloatTensor<B>;

    /// Update the given tensor with the value where the mask is true.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `mask` - The boolean mask to select with.
    /// * `value` - The value to assign to the selected elements.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to the given value.
    fn float_mask_fill(
        tensor: FloatTensor<B>,
        mask: BoolTensor<B>,
        value: FloatElem<B>,
    ) -> FloatTensor<B>;

    /// Equal comparison of two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_equal(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> BoolTensor<B>;

    /// Element-wise non-equality comparison.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_not_equal(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> BoolTensor<B> {
        let equal_tensor = B::float_equal(lhs, rhs);
        B::bool_not(equal_tensor)
    }

    /// Equal comparison of a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_equal_elem(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> BoolTensor<B>;

    /// Element-wise non-equality comparison with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_not_equal_elem(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> BoolTensor<B> {
        let equal_tensor = B::float_equal_elem(lhs, rhs);
        B::bool_not(equal_tensor)
    }

    /// Greater than comparison of two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_greater(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> BoolTensor<B>;

    /// Greater than comparison of a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_greater_elem(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> BoolTensor<B>;

    /// Greater than or equal comparison of two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_greater_equal(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> BoolTensor<B>;

    /// Greater than or equal comparison of a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_greater_equal_elem(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> BoolTensor<B>;

    /// Less than comparison of two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_lower(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> BoolTensor<B>;

    /// Less than comparison of a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_lower_elem(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> BoolTensor<B>;

    /// Less than or equal comparison of two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_lower_equal(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> BoolTensor<B>;

    /// Less than or equal comparison of a tensor and a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the result of the comparison.
    fn float_lower_equal_elem(lhs: FloatTensor<B>, rhs: FloatElem<B>) -> BoolTensor<B>;

    /// Detaches a tensor from the computation graph.
    fn float_detach(tensor: FloatTensor<B>) -> FloatTensor<B> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Sets the `require_grad` flag of a tensor.
    fn float_set_require_grad(tensor: FloatTensor<B>, _require_grad: bool) -> FloatTensor<B> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Returns the `require_grad` flag of a tensor.
    fn float_is_require_grad(_tensor: &FloatTensor<B>) -> bool {
        // Should only be overridden by autodiff backends.
        false
    }

    /// Sum of all elements in a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    ///
    /// # Returns
    ///
    /// A scalar tensor with the sum of all elements in `tensor`.
    fn float_sum(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Sum of all elements in a tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    /// * `dim` - The dimension along which to sum.
    ///
    /// # Returns
    ///
    /// A tensor with the sum of all elements in `tensor` along `dim`.
    fn float_sum_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>;

    /// Product of all elements in a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to product.
    ///
    /// # Returns
    ///
    /// A scalar tensor with the product of all elements in `tensor`.
    fn float_prod(tensor: FloatTensor<B>) -> FloatTensor<B> {
        // Product of all elements in a tensor
        B::float_exp(B::float_sum(B::float_log(tensor)))
    }

    /// Product of all elements in a tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to product.
    ///
    /// # Returns
    ///
    /// A tensor with the product of all elements in `tensor` along `dim`.
    fn float_prod_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        // Product of all elements in a tensor along a dimension
        B::float_exp(B::float_sum_dim(B::float_log(tensor), dim))
    }

    /// Mean of all elements in a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to mean.
    ///
    /// # Returns
    ///
    /// A scalar tensor with the mean of all elements in `tensor`.
    fn float_mean(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let num_elems = tensor.shape().num_elements();
        B::float_div_scalar(B::float_sum(tensor), (num_elems as i64).elem())
    }

    /// Mean of all elements in a tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to mean.
    /// * `dim` - The dimension along which to mean.
    ///
    /// # Returns
    ///
    /// A tensor with the mean of all elements in `tensor` along `dim`.
    fn float_mean_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>;

    /// Computes the cumulative sum of elements along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the cumulative sum of.
    /// * `dim` - The dimension along which to compute the cumulative sum.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape where each element is the cumulative sum
    /// of all elements up to and including that position along the dimension.
    fn float_cumsum(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B>;

    /// Converts a tensor to another floating point data type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to convert.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// A tensor with the same values as `tensor` but in the target floating point data type.
    fn float_cast(tensor: FloatTensor<B>, dtype: FloatDType) -> FloatTensor<B>;

    /// Returns a new tensor with exponential values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to exponentiate.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with exponential values.
    fn float_exp(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Returns a new tensor with natural logarithm values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the logarithm of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with natural logarithm values.
    fn float_log(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Returns a new tensor with logarithm values of (1 + Xi).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the logarithm of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with logarithm values of (1 + Xi).
    fn float_log1p(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Element-wise power with a FloatTensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of the elements of `rhs`.
    fn float_powf(lhs: FloatTensor<B>, rhs: FloatTensor<B>) -> FloatTensor<B>;

    /// Element-wise power with an IntTensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side floatTensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the value of `rhs`. Result is an IntTensor.
    fn float_powi(lhs: FloatTensor<B>, rhs: IntTensor<B>) -> FloatTensor<B> {
        Self::float_powf(lhs, B::int_into_float(rhs))
    }

    /// raises a tensor to the power of an int scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the value of `rhs`.
    fn float_powi_scalar(lhs: FloatTensor<B>, rhs: IntElem<B>) -> FloatTensor<B> {
        Self::float_powf_scalar(lhs, rhs.elem::<f32>())
    }

    /// Returns a new tensor with values raised to the power of float `value`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to exponentiate.
    /// * `value` - The exponent.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with values raised to the power of `value`.
    fn float_powf_scalar(tensor: FloatTensor<B>, value: f32) -> FloatTensor<B>;

    /// Returns a new tensor with square root values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the square root of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with square root values.
    fn float_sqrt(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Returns a new tensor with absolute values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take absolute value of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with absolute values.
    fn float_abs(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Returns a new tensor with cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the cosine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with cosine values.
    fn float_cos(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Returns a new tensor with sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the sine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with sine values.
    fn float_sin(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Returns a new tensor with tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the tangent of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with tangent values.
    fn float_tan(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let sin = B::float_sin(tensor.clone());
        let cos = B::float_cos(tensor);
        B::float_div(sin, cos)
    }

    /// Returns a new tensor with hyperbolic cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the hyperbolic cosine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic cosine values.
    fn float_cosh(tensor: FloatTensor<B>) -> FloatTensor<B> {
        // cosh = ( e^x + e^(-x) ) / 2
        let e_x = B::float_exp(tensor.clone());
        let e_neg_x = B::float_exp(B::float_neg(tensor));
        let num = B::float_add(e_x, e_neg_x); // e^x + e^(-x)
        B::float_div_scalar(num, 2.0.elem())
    }

    /// Returns a new tensor with hyperbolic sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the hyperbolic sine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic sine values.
    fn float_sinh(tensor: FloatTensor<B>) -> FloatTensor<B> {
        // sinh = ( e^x - e^(-x) ) / 2
        let e_x = B::float_exp(tensor.clone());
        let e_neg_x = B::float_exp(B::float_neg(tensor));
        let num = B::float_sub(e_x, e_neg_x); // e^x - e^(-x)
        B::float_div_scalar(num, 2.0.elem())
    }

    /// Returns a new tensor with hyperbolic tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the hyperbolic tangent of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with hyperbolic tangent values.
    fn float_tanh(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let sinh = B::float_sinh(tensor.clone());
        let cosh = B::float_cosh(tensor);
        B::float_div(sinh, cosh)
    }

    /// Returns a new tensor with rounded values.
    ///
    /// This function should implement the [round half to even](https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even)
    /// strategy, with halfway cases rounded to the nearest even integer value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to be rounded.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with rounded values.
    fn float_round(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Returns a new tensor with floored values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to be floored.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with floored values.
    fn float_floor(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Returns a new tensor with ceiled values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to be ceiled.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with ceiled values.
    fn float_ceil(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Returns a new tensor with the error function values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the error function of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with error function values.
    fn float_erf(tensor: FloatTensor<B>) -> FloatTensor<B>;

    /// Concatenates tensors along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors to concatenate.
    /// * `dim` - The dimension along which to concatenate.
    ///
    /// # Returns
    ///
    /// A tensor with the concatenated tensors along `dim`.
    fn float_cat(tensors: Vec<FloatTensor<B>>, dim: usize) -> FloatTensor<B> {
        cat_with_slice_assign::<B, Float>(
            tensors.into_iter().map(TensorPrimitive::Float).collect(),
            dim,
        )
        .tensor()
    }

    /// Gets the indices of the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    /// * `dim` - The dimension along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the indices of the maximum elements of `tensor` along `dim`.
    fn float_argmax(tensor: FloatTensor<B>, dim: usize) -> IntTensor<B>;

    /// Gets the indices of the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements of.
    /// * `dim` - The dimension along which to get the minimum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the indices of the minimum elements of `tensor` along `dim`.
    fn float_argmin(tensor: FloatTensor<B>, dim: usize) -> IntTensor<B>;

    /// Gets the maximum element of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    ///
    /// # Returns
    ///
    /// A tensor with the maximum element of `tensor`.
    fn float_max(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let shape = tensor.shape();
        let tensor = B::float_reshape(tensor, Shape::new([shape.num_elements()]));

        B::float_max_dim(tensor, 0)
    }

    /// Gets the maximum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    /// * `dim` - The dimension along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the maximum elements of `tensor` along `dim`.
    fn float_max_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        let index = B::float_argmax(tensor.clone(), dim);

        B::float_gather(dim, tensor, index)
    }

    /// Gets the maximum elements of a tensor along an axis and their indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    /// * `dim` - The dimension along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tuple with the maximum elements of `tensor` along `dim` and their indices.
    fn float_max_dim_with_indices(
        tensor: FloatTensor<B>,
        dim: usize,
    ) -> (FloatTensor<B>, IntTensor<B>) {
        let index = B::float_argmax(tensor.clone(), dim);
        let values = B::float_gather(dim, tensor, index.clone());

        (values, index)
    }

    /// Gets the minimum element of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements of.
    ///
    /// # Returns
    ///
    /// A tensor with the minimum element of `tensor`.
    fn float_min(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let shape = tensor.shape();
        let tensor = B::float_reshape(tensor, Shape::new([shape.num_elements()]));

        B::float_min_dim(tensor, 0)
    }

    /// Gets the minimum elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements of.
    /// * `dim` - The dimension along which to get the minimum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the minimum elements of `tensor` along `dim`.
    fn float_min_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        let index = B::float_argmin(tensor.clone(), dim);

        B::float_gather(dim, tensor, index)
    }

    /// Gets the minimum elements of a tensor along an axis and their indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements of.
    /// * `dim` - The dimension along which to get the minimum elements.
    ///
    /// # Returns
    ///
    /// A tuple with the minimum elements of `tensor` along `dim` and their indices.
    fn float_min_dim_with_indices(
        tensor: FloatTensor<B>,
        dim: usize,
    ) -> (FloatTensor<B>, IntTensor<B>) {
        let index = B::float_argmin(tensor.clone(), dim);
        let values = B::float_gather(dim, tensor, index.clone());

        (values, index)
    }

    /// Gets the maximum absolute element of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    ///
    /// # Returns
    ///
    /// A tensor with the maximum element of `tensor`.
    fn float_max_abs(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let shape = tensor.shape();
        let tensor = B::float_reshape(tensor, Shape::new([shape.num_elements()]));

        B::float_max_abs_dim(tensor, 0)
    }

    /// Gets the maximum absolute elements of a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    /// * `dim` - The dimension along which to get the maximum elements.
    ///
    /// # Returns
    ///
    /// A tensor with the maximum elements of `tensor` along `dim`.
    fn float_max_abs_dim(tensor: FloatTensor<B>, dim: usize) -> FloatTensor<B> {
        B::float_max_dim(B::float_abs(tensor), dim)
    }

    /// Tests if any element in the float `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if any element in the tensor is True, False otherwise.
    fn float_any(tensor: FloatTensor<B>) -> BoolTensor<B> {
        let bool_tensor = B::float_equal_elem(tensor, 0.0f32.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::float_sum(B::bool_into_float(bool_tensor));
        B::float_greater_elem(sum, 0.0f32.elem())
    }

    /// Tests if any element in the float `tensor` evaluates to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same size as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if any element along this dim in the
    /// input evaluates to True, False otherwise.
    fn float_any_dim(tensor: FloatTensor<B>, dim: usize) -> BoolTensor<B> {
        let bool_tensor = B::float_equal_elem(tensor, 0.0f32.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::float_sum_dim(B::bool_into_float(bool_tensor), dim);
        B::float_greater_elem(sum, 0.0f32.elem())
    }

    /// Tests if all elements in the float `tensor` evaluate to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, 1, Bool>` with a single element, True if all elements in the input tensor
    /// evaluate to True, False otherwise.
    fn float_all(tensor: FloatTensor<B>) -> BoolTensor<B> {
        let num_elems = tensor.shape().num_elements();
        let bool_tensor = B::float_equal_elem(tensor, 0.0f32.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::float_sum(B::bool_into_float(bool_tensor));
        B::float_equal_elem(sum, (num_elems as f32).elem())
    }

    /// Tests if all elements in the float `tensor` evaluate to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same size as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if all elements along this dim in the input
    /// evaluates to True, False otherwise.
    fn float_all_dim(tensor: FloatTensor<B>, dim: usize) -> BoolTensor<B> {
        let num_elems = tensor.shape().dims[dim];
        let bool_tensor = B::float_equal_elem(tensor, 0.0f32.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::float_sum_dim(B::bool_into_float(bool_tensor), dim);
        B::float_equal_elem(sum, (num_elems as f32).elem())
    }

    /// Returns the signs of the float `tensor`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to extract the signs from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` containing the signs of the elements of `tensor`.
    fn float_sign(tensor: FloatTensor<B>) -> FloatTensor<B> {
        let zeros = B::float_zeros(
            tensor.shape(),
            &B::float_device(&tensor),
            tensor.dtype().into(),
        );
        let less_than_zero = B::float_lower_elem(tensor.clone(), 0.0f32.elem());
        let greater_than_zero = B::float_greater_elem(tensor, 0.0f32.elem());

        let mut result = B::float_mask_fill(zeros, less_than_zero, (-1.0f32).elem());
        result = B::float_mask_fill(result, greater_than_zero, 1.0f32.elem());
        result
    }

    /// Broadcasts the float `tensor` to the given `shape`.
    fn float_expand(tensor: FloatTensor<B>, shape: Shape) -> FloatTensor<B>;

    /// Sort the elements of the input `tensor` by value in along a given dimension.
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
    fn float_sort(tensor: FloatTensor<B>, dim: usize, descending: bool) -> FloatTensor<B> {
        sort::<B, Float>(TensorPrimitive::Float(tensor), dim, descending).tensor()
    }

    /// Sort the elements of the input `tensor` by value in along a given dimension.
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
    fn float_sort_with_indices(
        tensor: FloatTensor<B>,
        dim: usize,
        descending: bool,
    ) -> (FloatTensor<B>, IntTensor<B>) {
        let (values, indices) =
            sort_with_indices::<B, Float>(TensorPrimitive::Float(tensor), dim, descending);
        (values.tensor(), indices)
    }

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
    fn float_argsort(tensor: FloatTensor<B>, dim: usize, descending: bool) -> IntTensor<B> {
        argsort::<B, Float>(TensorPrimitive::Float(tensor), dim, descending)
    }

    /// Samples tensor as a two-dimensional spatial grid of (possibly multi-channel) values,
    /// using the given locations in [-1, 1].
    ///
    /// Interpolation is bilinear.
    /// Padding is border: out of bounds locations will be clamped to the nearest border
    ///
    /// * `tensor` - The tensor being sampled from, shape (N, C, H_in, W_in)
    /// * `grid` - A tensor of locations, with shape (N, H_out, W_out, 2). Values are [-1, 1].
    ///   A [x = -1, y = -1] means top-left, and [x = 1, y = 1] means bottom-right
    /// * `method` - How to interpolate between samples
    ///
    /// # Returns
    ///
    /// A tensor with shape (N, C, H_out, W_out)
    fn float_grid_sample_2d(
        tensor: FloatTensor<B>,
        grid: FloatTensor<B>,
        method: InterpolateMode,
    ) -> FloatTensor<B> {
        match method {
            InterpolateMode::Bilinear => float_grid_sample_2d_bilinear::<B>(tensor, grid),
            _ => todo!("Default implementation for grid_sample_2d with {method:?} unimplemented"),
        }
    }

    /// Unfold windows along a dimension.
    ///
    /// Returns a view of the tensor with all complete windows of size `size` in dimension `dim`;
    /// where windows are advanced by `step` at each index.
    ///
    /// The number of windows is `max(0, (shape[dim] - size).ceil_div(step))`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor to unfold; of shape ``[pre=..., dim shape, post=...]``
    /// * `dim` - the selected dim.
    /// * `size` - the size of each unfolded window.
    /// * `step` - the step between each window.
    ///
    /// # Returns
    ///
    /// A tensor view with shape ``[pre=..., windows, size, post=...]``.
    fn float_unfold(tensor: FloatTensor<B>, dim: usize, size: usize, step: usize)
    -> FloatTensor<B>;

    /// Returns a new tensor with boolean elements indicating whether each element of the input is NaN.
    ///
    /// # Returns
    ///
    /// A boolean tensor where `true` indicates NaN and `false` indicates a non-NaN value.
    fn float_is_nan(tensor: FloatTensor<B>) -> BoolTensor<B> {
        // Check if the input tensor is NaN by comparing it to itself
        // NaN is the only value that is not equal to itself
        B::float_not_equal(tensor.clone(), tensor)
    }

    /// Returns a new tensor with boolean elements indicating whether each element of the input is infinite (either +INF or -INF).
    ///
    /// # Returns
    ///
    /// A boolean tensor where `true` indicates that the value is infinite
    fn float_is_inf(tensor: FloatTensor<B>) -> BoolTensor<B> {
        B::float_equal_elem(B::float_abs(tensor), f64::INFINITY.elem())
    }
}
