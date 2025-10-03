use super::cat::cat_with_slice_assign;
use super::repeat_dim::repeat_with_slice_assign;
use super::{BoolTensor, Device, FloatTensor, IntElem, IntTensor};
use crate::{
    Distribution, ElementConversion, Int, IntDType, TensorData, backend::Backend, tensor::Shape,
};
use crate::{TensorMetadata, argsort, sort, sort_with_indices};
use alloc::vec::Vec;
use core::ops::Range;

/// Int Tensor API for basic and numeric operations, see [tensor](crate::Tensor)
/// for documentation on each function.
pub trait IntTensorOps<B: Backend> {
    /// Creates a new int tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The integer tensor with the given shape.
    fn int_empty(shape: Shape, device: &Device<B>, dtype: IntDType) -> IntTensor<B>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn int_into_data(tensor: IntTensor<B>) -> impl Future<Output = TensorData> + Send;

    /// Creates a tensor from the data structure.
    ///
    /// # Arguments
    ///
    /// * `data` - The data structure.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the data.
    fn int_from_data(data: TensorData, device: &Device<B>) -> IntTensor<B>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn int_device(tensor: &IntTensor<B>) -> Device<B>;

    /// Moves the tensor to the given device.
    fn int_to_device(tensor: IntTensor<B>, device: &Device<B>) -> IntTensor<B>;

    /// Reshapes the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `shape` - The new shape.
    ///
    /// # Returns
    ///
    /// The tensor with the new shape.
    fn int_reshape(tensor: IntTensor<B>, shape: Shape) -> IntTensor<B>;

    /// Gets the element at the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `slices` - The slices specifying ranges and steps for each dimension.
    ///
    /// # Returns
    ///
    /// The elements at the given indices.
    fn int_slice(tensor: IntTensor<B>, slices: &[crate::Slice]) -> IntTensor<B>;

    /// Sets the values in the tensor for the given ranges.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `ranges` - The ranges to set the values for.
    ///
    /// # Returns
    ///
    /// The tensor with the values set for the given ranges.
    fn int_slice_assign(
        tensor: IntTensor<B>,
        slices: &[crate::Slice],
        value: IntTensor<B>,
    ) -> IntTensor<B>;

    /// Converts int tensor to float tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The int tensor with the same data as the float tensor.
    fn int_into_float(tensor: IntTensor<B>) -> FloatTensor<B>;

    /// Fills the tensor with values from the source tensor if the mask is true at the given
    /// indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `mask` - The mask.
    /// * `source` - The source tensor.
    ///
    /// # Returns
    ///
    /// The tensor with the values filled.
    fn int_mask_where(
        tensor: IntTensor<B>,
        mask: BoolTensor<B>,
        source: IntTensor<B>,
    ) -> IntTensor<B>;

    /// Fills the tensor with the given value if the mask is true at the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `mask` - The mask.
    /// * `value` - The value.
    ///
    /// # Returns
    ///
    /// The tensor with the values filled.
    fn int_mask_fill(tensor: IntTensor<B>, mask: BoolTensor<B>, value: IntElem<B>) -> IntTensor<B>;

    /// Gather elements from the tensor at the given indices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to gather from.
    /// * `tensor` - The tensor.
    /// * `indices` - The indices.
    fn int_gather(dim: usize, tensor: IntTensor<B>, indices: IntTensor<B>) -> IntTensor<B>;

    /// Scatter a given value to the tensor at the given indices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to scatter to.
    /// * `tensor` - The tensor.
    /// * `indices` - The indices.
    /// * `value` - The value.
    ///
    /// # Returns
    ///
    /// The tensor with the values scattered.
    fn int_scatter(
        dim: usize,
        tensor: IntTensor<B>,
        indices: IntTensor<B>,
        value: IntTensor<B>,
    ) -> IntTensor<B>;

    /// Select tensor elements along the given dimension corresponding to the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements.
    fn int_select(tensor: IntTensor<B>, dim: usize, indices: IntTensor<B>) -> IntTensor<B>;

    /// Assign the selected elements along the given dimension corresponding to the given indices
    /// to the given value.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices.
    /// * `value` - The value.
    ///
    /// # Returns
    ///
    /// The tensor with the selected elements assigned to the given value.
    fn int_select_assign(
        tensor: IntTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
        value: IntTensor<B>,
    ) -> IntTensor<B>;

    /// Repeats the tensor along the given dimension the given number of times.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to repeat.
    /// * `times` - The number of times to repeat.
    ///
    /// # Returns
    ///
    /// The tensor with the given dimension repeated the given number of times.
    fn int_repeat_dim(tensor: IntTensor<B>, dim: usize, times: usize) -> IntTensor<B> {
        repeat_with_slice_assign::<B, Int>(tensor, dim, times)
    }

    /// Concatenates the given tensors along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors.
    /// * `dim` - The dimension to concatenate along.
    ///
    /// # Returns
    ///
    /// The concatenated tensor.
    fn int_cat(tensors: Vec<IntTensor<B>>, dim: usize) -> IntTensor<B> {
        cat_with_slice_assign::<B, Int>(tensors, dim)
    }

    /// Element-wise equality comparison.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_equal(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B>;

    /// Element-wise non-equality comparison.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_not_equal(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B> {
        let equal_tensor = B::int_equal(lhs, rhs);
        B::bool_not(equal_tensor)
    }

    /// Element-wise equality comparison with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_equal_elem(lhs: IntTensor<B>, rhs: IntElem<B>) -> BoolTensor<B>;

    /// Element-wise non-equality comparison with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_not_equal_elem(lhs: IntTensor<B>, rhs: IntElem<B>) -> BoolTensor<B> {
        let equal_tensor = B::int_equal_elem(lhs, rhs);
        B::bool_not(equal_tensor)
    }

    /// Element-wise greater than comparison.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_greater(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B>;

    /// Element-wise greater than comparison with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_greater_elem(lhs: IntTensor<B>, rhs: IntElem<B>) -> BoolTensor<B>;

    /// Element-wise greater than or equal comparison.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_greater_equal(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B>;

    /// Element-wise greater than or equal comparison with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_greater_equal_elem(lhs: IntTensor<B>, rhs: IntElem<B>) -> BoolTensor<B>;

    /// Element-wise less than comparison.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_lower(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B>;

    /// Element-wise less than comparison with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_lower_elem(lhs: IntTensor<B>, rhs: IntElem<B>) -> BoolTensor<B>;

    /// Element-wise less than or equal comparison.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_lower_equal(lhs: IntTensor<B>, rhs: IntTensor<B>) -> BoolTensor<B>;

    /// Element-wise less than or equal comparison with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the result of the comparison.
    fn int_lower_equal_elem(lhs: IntTensor<B>, rhs: IntElem<B>) -> BoolTensor<B>;

    // ====  NUMERIC ==== //

    /// Element-wise addition.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of the addition.
    fn int_add(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Element-wise addition with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of the addition.
    fn int_add_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

    /// Element-wise power with a IntTensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side IntTensor.
    /// * `rhs` - The right hand side IntTensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of the elements of `rhs`.
    fn int_powi(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B> {
        B::float_into_int(B::float_powi(B::int_into_float(lhs), rhs))
    }

    /// Element-wise power with a floatTensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side floatTensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the value of `rhs`. Result is an IntTensor.
    fn int_powf(lhs: IntTensor<B>, rhs: FloatTensor<B>) -> IntTensor<B> {
        B::float_into_int(B::float_powf(B::int_into_float(lhs), rhs))
    }

    /// Element-wise power with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the value of `rhs`.
    fn int_powi_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B> {
        B::float_into_int(B::float_powi_scalar(B::int_into_float(lhs), rhs))
    }

    /// Element-wise power with a floatTensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the value of `rhs`. Result is an IntTensor.
    fn int_powf_scalar(lhs: IntTensor<B>, rhs: f32) -> IntTensor<B> {
        B::float_into_int(B::float_powf_scalar(B::int_into_float(lhs), rhs))
    }

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
    fn int_clamp_min(tensor: IntTensor<B>, min: IntElem<B>) -> IntTensor<B> {
        let mask = Self::int_lower_elem(tensor.clone(), min);
        Self::int_mask_fill(tensor, mask, min)
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
    fn int_clamp_max(tensor: IntTensor<B>, max: IntElem<B>) -> IntTensor<B> {
        let mask = Self::int_greater_elem(tensor.clone(), max);
        Self::int_mask_fill(tensor, mask, max)
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
    fn int_clamp(tensor: IntTensor<B>, min: IntElem<B>, max: IntElem<B>) -> IntTensor<B> {
        Self::int_clamp_min(Self::int_clamp_max(tensor, max), min)
    }

    /// Element-wise subtraction.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of the subtraction.
    fn int_sub(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Element-wise subtraction with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of the subtraction.
    fn int_sub_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

    /// Element-wise multiplication.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of the multiplication.
    fn int_mul(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Element-wise multiplication with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of the multiplication.
    fn int_mul_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

    /// Element-wise division.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The result of the division.
    fn int_div(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Element-wise division with a scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of the division.
    fn int_div_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

    /// Element-wise modulus.
    ///
    /// # Arguments
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of applying the modulus of the scalar to the tensor.
    fn int_remainder(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Element-wise modulus with a scalar.
    ///
    /// # Arguments
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of applying the modulus of the scalar to the tensor.
    fn int_remainder_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

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
    fn int_matmul(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Element-wise negation.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to negate.
    ///
    /// # Returns
    ///
    /// The negated tensor.
    fn int_neg(tensor: IntTensor<B>) -> IntTensor<B> {
        Self::int_mul_scalar(tensor, (-1.0).elem::<IntElem<B>>())
    }

    /// Creates a tensor of zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The tensor of zeros.
    fn int_zeros(shape: Shape, device: &Device<B>, dtype: IntDType) -> IntTensor<B> {
        Self::int_from_data(TensorData::full_dtype(shape, 0, dtype.into()), device)
    }

    /// Creates a tensor of ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The tensor of ones.
    fn int_ones(shape: Shape, device: &Device<B>, dtype: IntDType) -> IntTensor<B> {
        Self::int_from_data(TensorData::full_dtype(shape, 1, dtype.into()), device)
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
    fn int_full(
        shape: Shape,
        fill_value: IntElem<B>,
        device: &Device<B>,
        dtype: IntDType,
    ) -> IntTensor<B> {
        Self::int_from_data(
            TensorData::full_dtype(shape, fill_value, dtype.into()),
            device,
        )
    }

    /// Sums all elements in the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    ///
    /// # Returns
    ///
    /// The sum of all elements in the tensor.
    fn int_sum(tensor: IntTensor<B>) -> IntTensor<B>;

    /// Sums all elements in the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to sum.
    /// * `dim` - The dimension to sum along.
    ///
    /// # Returns
    ///
    /// The sum of all elements in the tensor along the dimension.
    fn int_sum_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>;

    /// Computes the product of all elements in the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the product of.
    ///
    /// # Returns
    ///
    /// The product of all elements in the tensor.
    fn int_prod(tensor: IntTensor<B>) -> IntTensor<B>;

    /// Computes the product of all elements in the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the product of.
    /// * `dim` - The dimension to compute the product along.
    ///
    /// # Returns
    ///
    /// The product of all elements in the tensor along the dimension.
    fn int_prod_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>;

    /// Computes the mean of all elements in the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    ///
    /// # Returns
    ///
    /// The mean of all elements in the tensor.
    fn int_mean(tensor: IntTensor<B>) -> IntTensor<B> {
        let num_elems = tensor.shape().num_elements();
        B::int_div_scalar(B::int_sum(tensor), (num_elems as i64).elem())
    }

    /// Computes the mean of all elements in the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    ///
    /// # Returns
    ///
    /// The mean of all elements in the tensor along the dimension.
    fn int_mean_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>;

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
    fn int_cumsum(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>;

    /// Gets the indices of the maximum elements along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum indices of.
    /// * `dim` - The dimension to get the maximum indices along.
    ///
    /// # Returns
    ///
    /// The indices of the maximum elements along the dimension.
    fn int_argmax(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>;

    /// Gets the indices of the minimum elements along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum indices of.
    /// * `dim` - The dimension to get the minimum indices along.
    ///
    /// # Returns
    ///
    /// The indices of the minimum elements along the dimension.
    fn int_argmin(tensor: IntTensor<B>, dim: usize) -> IntTensor<B>;

    /// Gets the maximum element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum element of.
    ///
    /// # Returns
    ///
    /// The maximum element in the tensor.
    fn int_max(tensor: IntTensor<B>) -> IntTensor<B> {
        let shape = tensor.shape();
        let tensor = B::int_reshape(tensor, Shape::new([shape.num_elements()]));

        B::int_max_dim(tensor, 0)
    }

    /// Gets the maximum element in the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum element of.
    /// * `dim` - The dimension to get the maximum element along.
    ///
    /// # Returns
    ///
    /// The maximum element in the tensor along the dimension.
    fn int_max_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        let index = B::int_argmax(tensor.clone(), dim);
        B::int_gather(dim, tensor, index)
    }

    /// Gets the maximum elements and corresponding indices along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements and indices of.
    /// * `dim` - The dimension to get the maximum elements and indices along.
    ///
    /// # Returns
    ///
    /// The maximum elements and corresponding indices along the dimension.
    fn int_max_dim_with_indices(tensor: IntTensor<B>, dim: usize) -> (IntTensor<B>, IntTensor<B>) {
        let index = B::int_argmax(tensor.clone(), dim);
        let values = B::int_gather(dim, tensor, index.clone());

        (values, index)
    }

    /// Gets the maximum absolute element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum element of.
    ///
    /// # Returns
    ///
    /// The maximum element in the tensor.
    fn int_max_abs(tensor: IntTensor<B>) -> IntTensor<B> {
        let shape = tensor.shape();
        let tensor = B::int_reshape(tensor, Shape::new([shape.num_elements()]));

        B::int_max_abs_dim(tensor, 0)
    }

    /// Gets the maximum absolute element in the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum element of.
    /// * `dim` - The dimension to get the maximum element along.
    ///
    /// # Returns
    ///
    /// The maximum element in the tensor along the dimension.
    fn int_max_abs_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        B::int_max_dim(B::int_abs(tensor), dim)
    }

    /// Gets the minimum element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum element of.
    ///
    /// # Returns
    ///
    /// The minimum element in the tensor.
    fn int_min(tensor: IntTensor<B>) -> IntTensor<B> {
        let shape = tensor.shape();
        let tensor = B::int_reshape(tensor, Shape::new([shape.num_elements()]));

        B::int_min_dim(tensor, 0)
    }

    /// Gets the minimum elements in the tensor along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum element of.
    /// * `dim` - The dimension to get the minimum element along.
    ///
    /// # Returns
    ///
    /// The minimum element in the tensor along the dimension.
    fn int_min_dim(tensor: IntTensor<B>, dim: usize) -> IntTensor<B> {
        let index = B::int_argmin(tensor.clone(), dim);
        B::int_gather(dim, tensor, index)
    }

    /// Gets the minimum elements and corresponding indices along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the minimum elements and indices of.
    /// * `dim` - The dimension to get the minimum elements and indices along.
    ///
    /// # Returns
    ///
    /// The minimum elements and corresponding indices along the dimension.
    fn int_min_dim_with_indices(tensor: IntTensor<B>, dim: usize) -> (IntTensor<B>, IntTensor<B>) {
        let indices = B::int_argmin(tensor.clone(), dim);
        let values = B::int_gather(dim, tensor, indices.clone());

        (values, indices)
    }

    /// Returns a new tensor with absolute values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take absolute value of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with absolute values.
    fn int_abs(tensor: IntTensor<B>) -> IntTensor<B>;

    /// Transposes an int tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn int_transpose(tensor: IntTensor<B>) -> IntTensor<B> {
        let ndims = tensor.shape().num_dims();
        Self::int_swap_dims(tensor, ndims - 2, ndims - 1)
    }

    /// Swaps two dimensions of an int tensor.
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
    fn int_swap_dims(tensor: IntTensor<B>, dim1: usize, dim2: usize) -> IntTensor<B>;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn int_permute(tensor: IntTensor<B>, axes: &[usize]) -> IntTensor<B>;

    /// Reverse the order of elements in a tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reverse.
    /// * `axes` - The axes to reverse.
    ///
    /// The tensor with the elements reversed.
    fn int_flip(tensor: IntTensor<B>, axes: &[usize]) -> IntTensor<B>;

    /// Creates a new int tensor with random values.
    ///
    ///  # Arguments
    ///  * `shape` - The shape of the tensor.
    ///  * `distribution` - The distribution to sample from.
    ///  * `device` - The device to create the tensor on.
    ///
    ///  # Returns
    ///
    ///  The tensor with the given shape and random values.
    fn int_random(shape: Shape, distribution: Distribution, device: &Device<B>) -> IntTensor<B>;

    /// Creates a new tensor with values from the given range with the given step size.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values.
    /// * `step` - The step size.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given values.
    fn int_arange_step(range: Range<i64>, step: usize, device: &Device<B>) -> IntTensor<B> {
        let value = range
            .step_by(step)
            .map(|i| i.elem())
            .collect::<Vec<IntElem<B>>>();
        let shape = Shape::new([value.len()]);
        let data = TensorData::new(value, shape);
        B::int_from_data(data, device)
    }

    /// Creates a new tensor with values from the given range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given values.
    ///
    /// # Remarks
    ///
    /// Uses `arange_step` with a step size of 1 under the hood.
    fn int_arange(range: Range<i64>, device: &Device<B>) -> IntTensor<B> {
        Self::int_arange_step(range, 1, device)
    }

    /// Tests if any element in the int `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if any element in the tensor is True, False otherwise.
    fn int_any(tensor: IntTensor<B>) -> BoolTensor<B> {
        let bool_tensor = B::int_equal_elem(tensor, 0.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::int_sum(B::bool_into_int(bool_tensor));
        B::int_greater_elem(sum, 0.elem())
    }

    /// Tests if any element in the int `tensor` evaluates to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same size as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if any element along this dim in the input
    /// evaluates to True, False otherwise.
    fn int_any_dim(tensor: IntTensor<B>, dim: usize) -> BoolTensor<B> {
        let bool_tensor = B::int_equal_elem(tensor, 0.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::int_sum_dim(B::bool_into_int(bool_tensor), dim);
        B::int_greater_elem(sum, 0.elem())
    }

    /// Tests if all elements in the int `tensor` evaluate to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, 1, Bool>` with a single element, True if all elements in the input tensor
    /// evaluate to True, False otherwise.
    fn int_all(tensor: IntTensor<B>) -> BoolTensor<B> {
        let num_elems = tensor.shape().num_elements();
        let bool_tensor = B::int_equal_elem(tensor, 0.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::int_sum(B::bool_into_int(bool_tensor));
        B::int_equal_elem(sum, (num_elems as i32).elem())
    }

    /// Tests if all elements in the int `tensor` evaluate to True along a given dimension `dim`.
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
    fn int_all_dim(tensor: IntTensor<B>, dim: usize) -> BoolTensor<B> {
        let num_elems = tensor.shape().dims[dim];
        let bool_tensor = B::int_equal_elem(tensor, 0.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::int_sum_dim(B::bool_into_int(bool_tensor), dim);
        B::int_equal_elem(sum, (num_elems as i32).elem())
    }

    /// Returns the signs of the int `tensor`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to extract the signs from.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` containing the signs of the elements of `tensor`.
    fn int_sign(tensor: IntTensor<B>) -> IntTensor<B> {
        let dtype = tensor.dtype();
        let zeros = B::int_zeros(tensor.shape(), &B::int_device(&tensor), dtype.into());
        let less_than_zero = B::int_lower_elem(tensor.clone(), 0.0f32.elem());
        let greater_than_zero = B::int_greater_elem(tensor, 0.0f32.elem());

        let mut result = B::int_mask_fill(zeros, less_than_zero, (-1.0f32).elem());
        result = B::int_mask_fill(result, greater_than_zero, 1.0f32.elem());
        result
    }

    /// Broadcasts the int `tensor` to the given `shape`.
    fn int_expand(tensor: IntTensor<B>, shape: Shape) -> IntTensor<B>;

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
    fn int_sort(tensor: IntTensor<B>, dim: usize, descending: bool) -> IntTensor<B> {
        sort::<B, Int>(tensor, dim, descending)
    }

    /// Sort the elements of the input `tensor` by value along a given dimension.
    ///
    /// This sort is unstable (i.e., may reorder equal elements).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    /// * `dim` - The axis along which to sort.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor and corresponding indices, where
    /// the elements are sorted by value and the indices map back to the original input tensor.
    fn int_sort_with_indices(
        tensor: IntTensor<B>,
        dim: usize,
        descending: bool,
    ) -> (IntTensor<B>, IntTensor<B>) {
        sort_with_indices::<B, Int>(tensor, dim, descending)
    }

    /// Returns the indices that sort the elements of the input `tensor` by value
    /// along a given dimension.
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
    fn int_argsort(tensor: IntTensor<B>, dim: usize, descending: bool) -> IntTensor<B> {
        argsort::<B, Int>(tensor, dim, descending)
    }

    /// Bitwise AND operation for Int Tensors
    fn bitwise_and(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Bitwise AND operation for Int Tensors with a scalar
    fn bitwise_and_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

    /// Bitwise OR operation for Int Tensors
    fn bitwise_or(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Bitwise OR operation for Int Tensors with a scalar
    fn bitwise_or_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

    /// Bitwise XOR operation for Int Tensors
    fn bitwise_xor(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Bitwise XOR operation for Int Tensors with a scalar
    fn bitwise_xor_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

    /// Bitwise NOT operation for Int Tensors
    fn bitwise_not(tensor: IntTensor<B>) -> IntTensor<B>;

    /// Bitwise left shift operation for Int Tensors
    fn bitwise_left_shift(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Bitwise left shift operation for Int Tensors with a scalar
    fn bitwise_left_shift_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

    /// Bitwise right shift operation for Int Tensors
    fn bitwise_right_shift(lhs: IntTensor<B>, rhs: IntTensor<B>) -> IntTensor<B>;

    /// Bitwise right shift operation for Int Tensors with a scalar
    fn bitwise_right_shift_scalar(lhs: IntTensor<B>, rhs: IntElem<B>) -> IntTensor<B>;

    /// Converts a tensor to another integer data type.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to convert.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// A tensor with the same values as `tensor` but in the target integer data type.
    fn int_cast(tensor: IntTensor<B>, dtype: IntDType) -> IntTensor<B>;

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
    fn int_unfold(tensor: IntTensor<B>, dim: usize, size: usize, step: usize) -> IntTensor<B>;
}
