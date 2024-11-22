use super::{cat::cat_with_slice_assign, IntTensor};
use super::{repeat_dim::repeat_with_slice_assign, ByteElem};
use super::{BoolTensor, ByteTensor, Device, FloatTensor};
use crate::tensor::api::{chunk, narrow, split, split_with_sizes};
use crate::{backend::Backend, tensor::Shape, Distribution, ElementConversion, TensorData};
use crate::{cast::ToElement, Byte};
use alloc::vec::Vec;
use core::future::Future;
use core::ops::Range;

use crate::{argsort, sort, sort_with_indices, TensorMetadata};

/// Int Tensor API for basic and numeric operations, see [tensor](crate::Tensor)
/// for documentation on each function.
pub trait ByteTensorOps<B: Backend> {
    /// Creates a new int tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The integer tensor with the given shape.
    fn byte_empty(shape: Shape, device: &Device<B>) -> ByteTensor<B>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn byte_into_data(tensor: ByteTensor<B>) -> impl Future<Output = TensorData> + 'static + Send;

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
    fn byte_from_data(data: TensorData, device: &Device<B>) -> ByteTensor<B>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn byte_device(tensor: &ByteTensor<B>) -> Device<B>;

    /// Moves the tensor to the given device.
    fn byte_to_device(tensor: ByteTensor<B>, device: &Device<B>) -> ByteTensor<B>;

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
    fn byte_reshape(tensor: ByteTensor<B>, shape: Shape) -> ByteTensor<B>;

    /// Gets the element at the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `indices` - The indices.
    ///
    /// # Returns
    ///
    /// The elements at the given indices.
    fn byte_slice(tensor: ByteTensor<B>, indices: &[Range<usize>]) -> ByteTensor<B>;

    /// Sets the element at the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `indices` - The indices.
    ///
    /// # Returns
    ///
    /// The tensor with the element at the given indices set.
    fn byte_slice_assign(
        tensor: ByteTensor<B>,
        indices: &[Range<usize>],
        value: ByteTensor<B>,
    ) -> ByteTensor<B>;

    /// Converts byte tensor to float tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The float tensor with the same data as the byte tensor.
    fn byte_into_float(tensor: ByteTensor<B>) -> FloatTensor<B>;

    /// Converts byte tensor to int tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The int tensor with the same data as the byte tensor.
    fn byte_into_int(tensor: ByteTensor<B>) -> IntTensor<B>;

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
    fn byte_mask_where(
        tensor: ByteTensor<B>,
        mask: BoolTensor<B>,
        source: ByteTensor<B>,
    ) -> ByteTensor<B>;

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
    fn byte_mask_fill(
        tensor: ByteTensor<B>,
        mask: BoolTensor<B>,
        value: ByteElem<B>,
    ) -> ByteTensor<B>;

    /// Gather elements from the tensor at the given indices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to gather from.
    /// * `tensor` - The tensor.
    /// * `indices` - The indices.
    fn byte_gather(dim: usize, tensor: ByteTensor<B>, indices: IntTensor<B>) -> ByteTensor<B>;

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
    fn byte_scatter(
        dim: usize,
        tensor: ByteTensor<B>,
        indices: IntTensor<B>,
        value: ByteTensor<B>,
    ) -> ByteTensor<B>;

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
    fn byte_select(tensor: ByteTensor<B>, dim: usize, indices: IntTensor<B>) -> ByteTensor<B>;

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
    fn byte_select_assign(
        tensor: ByteTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
        value: ByteTensor<B>,
    ) -> ByteTensor<B>;

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
    fn byte_repeat_dim(tensor: ByteTensor<B>, dim: usize, times: usize) -> ByteTensor<B> {
        repeat_with_slice_assign::<B, Byte>(tensor, dim, times)
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
    fn byte_cat(tensors: Vec<ByteTensor<B>>, dim: usize) -> ByteTensor<B> {
        cat_with_slice_assign::<B, Byte>(tensors, dim)
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
    fn byte_equal(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B>;

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
    fn byte_not_equal(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B> {
        let equal_tensor = B::byte_equal(lhs, rhs);
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
    fn byte_equal_elem(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> BoolTensor<B>;

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
    fn byte_not_equal_elem(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> BoolTensor<B> {
        let equal_tensor = B::byte_equal_elem(lhs, rhs);
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
    fn byte_greater(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B>;

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
    fn byte_greater_elem(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> BoolTensor<B>;

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
    fn byte_greater_equal(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B>;

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
    fn byte_greater_equal_elem(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> BoolTensor<B>;

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
    fn byte_lower(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B>;

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
    fn byte_lower_elem(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> BoolTensor<B>;

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
    fn byte_lower_equal(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> BoolTensor<B>;

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
    fn byte_lower_equal_elem(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> BoolTensor<B>;

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
    fn byte_add(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B>;

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
    fn byte_add_scalar(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> ByteTensor<B>;

    /// Element-wise power with a ByteTensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side ByteTensor.
    /// * `rhs` - The right hand side ByteTensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of the elements of `rhs`.
    fn byte_powi(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B> {
        B::float_into_byte(B::float_powf(
            B::byte_into_float(lhs),
            B::byte_into_float(rhs),
        ))
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
    /// The elements of `lhs` raised to the value of `rhs`. Result is an ByteTensor.
    fn byte_powf(lhs: ByteTensor<B>, rhs: FloatTensor<B>) -> ByteTensor<B> {
        B::float_into_byte(B::float_powf(B::byte_into_float(lhs), rhs))
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
    fn byte_powi_scalar(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> ByteTensor<B> {
        B::float_into_byte(B::float_powf_scalar(B::byte_into_float(lhs), rhs.to_f32()))
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
    /// The elements of `lhs` raised to the value of `rhs`. Result is an ByteTensor.
    fn byte_powf_scalar(lhs: ByteTensor<B>, rhs: f32) -> ByteTensor<B> {
        B::float_into_byte(B::float_powf_scalar(B::byte_into_float(lhs), rhs))
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
    fn byte_clamp_min(tensor: ByteTensor<B>, min: ByteElem<B>) -> ByteTensor<B> {
        let mask = Self::byte_lower_elem(tensor.clone(), min);
        Self::byte_mask_fill(tensor, mask, min)
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
    fn byte_clamp_max(tensor: ByteTensor<B>, max: ByteElem<B>) -> ByteTensor<B> {
        let mask = Self::byte_greater_elem(tensor.clone(), max);
        Self::byte_mask_fill(tensor, mask, max)
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
    fn byte_clamp(tensor: ByteTensor<B>, min: ByteElem<B>, max: ByteElem<B>) -> ByteTensor<B> {
        Self::byte_clamp_min(Self::byte_clamp_max(tensor, max), min)
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
    fn byte_sub(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B>;

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
    fn byte_sub_scalar(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> ByteTensor<B>;

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
    fn byte_mul(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B>;

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
    fn byte_mul_scalar(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> ByteTensor<B>;

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
    fn byte_div(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B>;

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
    fn byte_div_scalar(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> ByteTensor<B>;

    /// Element-wise modulus.
    ///
    /// # Arguments
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of applying the modulus of the scalar to the tensor.
    fn byte_remainder(lhs: ByteTensor<B>, rhs: ByteTensor<B>) -> ByteTensor<B>;

    /// Element-wise modulus with a scalar.
    ///
    /// # Arguments
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of applying the modulus of the scalar to the tensor.
    fn byte_remainder_scalar(lhs: ByteTensor<B>, rhs: ByteElem<B>) -> ByteTensor<B>;

    /// Element-wise negation.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to negate.
    ///
    /// # Returns
    ///
    /// The negated tensor.
    fn byte_neg(tensor: ByteTensor<B>) -> ByteTensor<B> {
        Self::byte_mul_scalar(tensor, (-1.0).elem::<ByteElem<B>>())
    }

    /// Creates a tensor of zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor of zeros.
    fn byte_zeros(shape: Shape, device: &Device<B>) -> ByteTensor<B>;

    /// Creates a tensor of ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor of ones.
    fn byte_ones(shape: Shape, device: &Device<B>) -> ByteTensor<B>;

    /// Creates a tensor filled with given value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `fill_value` - The value with which to fill the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor filled with given value
    fn byte_full(shape: Shape, fill_value: ByteElem<B>, device: &Device<B>) -> ByteTensor<B> {
        Self::byte_add_scalar(Self::byte_zeros(shape, device), fill_value)
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
    fn byte_sum(tensor: ByteTensor<B>) -> ByteTensor<B>;

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
    fn byte_sum_dim(tensor: ByteTensor<B>, dim: usize) -> ByteTensor<B>;

    /// Computes the product of all elements in the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the product of.
    ///
    /// # Returns
    ///
    /// The product of all elements in the tensor.
    fn byte_prod(tensor: ByteTensor<B>) -> ByteTensor<B>;

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
    fn byte_prod_dim(tensor: ByteTensor<B>, dim: usize) -> ByteTensor<B>;

    /// Computes the mean of all elements in the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to compute the mean of.
    ///
    /// # Returns
    ///
    /// The mean of all elements in the tensor.
    fn byte_mean(tensor: ByteTensor<B>) -> ByteTensor<B> {
        let num_elems = tensor.shape().num_elements();
        B::byte_div_scalar(B::byte_sum(tensor), (num_elems as i64).elem())
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
    fn byte_mean_dim(tensor: ByteTensor<B>, dim: usize) -> ByteTensor<B>;

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
    fn byte_argmax(tensor: ByteTensor<B>, dim: usize) -> IntTensor<B>;

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
    fn byte_argmin(tensor: ByteTensor<B>, dim: usize) -> IntTensor<B>;

    /// Gets the maximum element in the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum element of.
    ///
    /// # Returns
    ///
    /// The maximum element in the tensor.
    fn byte_max(tensor: ByteTensor<B>) -> ByteTensor<B> {
        let shape = tensor.shape();
        let tensor = B::byte_reshape(tensor, Shape::new([shape.num_elements()]));

        B::byte_max_dim(tensor, 0)
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
    fn byte_max_dim(tensor: ByteTensor<B>, dim: usize) -> ByteTensor<B> {
        let index = B::byte_argmax(tensor.clone(), dim);
        let ndim = tensor.shape().num_dims();

        B::byte_gather(ndim - 1, tensor, index)
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
    fn byte_max_dim_with_indices(
        tensor: ByteTensor<B>,
        dim: usize,
    ) -> (ByteTensor<B>, IntTensor<B>) {
        let index = B::byte_argmax(tensor.clone(), dim);
        let values = B::byte_gather(dim, tensor, index.clone());

        (values, index)
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
    fn byte_min(tensor: ByteTensor<B>) -> ByteTensor<B> {
        let shape = tensor.shape();
        let tensor = B::byte_reshape(tensor, Shape::new([shape.num_elements()]));

        B::byte_min_dim(tensor, 0)
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
    fn byte_min_dim(tensor: ByteTensor<B>, dim: usize) -> ByteTensor<B> {
        let index = B::byte_argmin(tensor.clone(), dim);
        let ndim = tensor.shape().num_dims();

        B::byte_gather(ndim - 1, tensor, index)
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
    fn byte_min_dim_with_indices(
        tensor: ByteTensor<B>,
        dim: usize,
    ) -> (ByteTensor<B>, IntTensor<B>) {
        let indices = B::byte_argmin(tensor.clone(), dim);
        let ndim = tensor.shape().num_dims();
        let values = B::byte_gather(ndim - 1, tensor, indices.clone());

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
    fn byte_abs(tensor: ByteTensor<B>) -> ByteTensor<B>;

    /// Transposes an int tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn byte_transpose(tensor: ByteTensor<B>) -> ByteTensor<B> {
        let ndims = tensor.shape().num_dims();
        Self::byte_swap_dims(tensor, ndims - 2, ndims - 1)
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
    fn byte_swap_dims(tensor: ByteTensor<B>, dim1: usize, dim2: usize) -> ByteTensor<B>;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn byte_permute(tensor: ByteTensor<B>, axes: &[usize]) -> ByteTensor<B>;

    /// Reverse the order of elements in a tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reverse.
    /// * `axes` - The axes to reverse.
    ///
    /// The tensor with the elements reversed.
    fn byte_flip(tensor: ByteTensor<B>, axes: &[usize]) -> ByteTensor<B>;

    /// Returns a new tensor with the given dimension narrowed to the given range.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which the tensor will be narrowed.
    /// * `start` - The starting point of the given range.
    /// * `length` - The ending point of the given range.
    /// # Panics
    ///
    /// - If the dimension is greater than the number of dimensions of the tensor.
    /// - If the given range exceeds the number of elements on the given dimension.
    ///
    /// # Returns
    ///
    /// A new tensor with the given dimension narrowed to the given range.
    fn byte_narrow(
        tensor: ByteTensor<B>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> ByteTensor<B> {
        narrow::<B, Byte>(tensor, dim, start, length)
    }

    /// Split the tensor along the given dimension into chunks.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `chunks` - The number of chunks to be produced.
    /// * `times` - The dimension along which the tensor will be split.
    ///
    /// # Returns
    ///
    /// A vector of tensors
    fn byte_chunk(tensor: ByteTensor<B>, chunks: usize, dim: usize) -> Vec<ByteTensor<B>> {
        chunk::<B, Byte>(tensor, chunks, dim)
    }

    /// Split the tensor along the given dimension into chunks of `split_size`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `split_size` - The size of a single chunk.
    /// * `times` - The dimension along which the tensor will be split.
    ///
    /// # Returns
    ///
    /// A vector of tensors.
    fn byte_split(tensor: ByteTensor<B>, split_size: usize, dim: usize) -> Vec<ByteTensor<B>> {
        split::<B, Byte>(tensor, split_size, dim)
    }

    /// Split the tensor along the given dimension into chunks with sizes in
    /// `dim` according to `split_sizes`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `split_sizes` - Vector of sizes for each chunk.
    /// * `times` - The dimension along which the tensor will be split.
    ///
    /// # Returns
    ///
    /// A vector of tensors.
    fn byte_split_with_sizes(
        tensor: ByteTensor<B>,
        split_sizes: Vec<usize>,
        dim: usize,
    ) -> Vec<ByteTensor<B>> {
        split_with_sizes::<B, Byte>(tensor, split_sizes, dim)
    }

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
    fn byte_random(shape: Shape, distribution: Distribution, device: &Device<B>) -> ByteTensor<B>;

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
    fn byte_arange_step(range: Range<i64>, step: usize, device: &Device<B>) -> ByteTensor<B> {
        let value = range
            .step_by(step)
            .map(|i| i.elem())
            .collect::<Vec<ByteElem<B>>>();
        let shape = Shape::new([value.len()]);
        let data = TensorData::new(value, shape);
        B::byte_from_data(data, device)
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
    fn byte_arange(range: Range<i64>, device: &Device<B>) -> ByteTensor<B> {
        Self::byte_arange_step(range, 1, device)
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
    fn byte_any(tensor: ByteTensor<B>) -> BoolTensor<B> {
        let bool_tensor = B::byte_equal_elem(tensor, 0.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::byte_sum(B::bool_into_byte(bool_tensor));
        B::byte_greater_elem(sum, 0.elem())
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
    fn byte_any_dim(tensor: ByteTensor<B>, dim: usize) -> BoolTensor<B> {
        let bool_tensor = B::byte_equal_elem(tensor, 0.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::byte_sum_dim(B::bool_into_byte(bool_tensor), dim);
        B::byte_greater_elem(sum, 0.elem())
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
    fn byte_all(tensor: ByteTensor<B>) -> BoolTensor<B> {
        let num_elems = tensor.shape().num_elements();
        let bool_tensor = B::byte_equal_elem(tensor, 0.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::byte_sum(B::bool_into_byte(bool_tensor));
        B::byte_equal_elem(sum, (num_elems as i32).elem())
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
    fn byte_all_dim(tensor: ByteTensor<B>, dim: usize) -> BoolTensor<B> {
        let num_elems = tensor.shape().dims[dim];
        let bool_tensor = B::byte_equal_elem(tensor, 0.elem());
        let bool_tensor = B::bool_not(bool_tensor);
        let sum = B::byte_sum_dim(B::bool_into_byte(bool_tensor), dim);
        B::byte_equal_elem(sum, (num_elems as i32).elem())
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
    fn byte_sign(tensor: ByteTensor<B>) -> ByteTensor<B> {
        let zeros = B::byte_zeros(tensor.shape(), &B::byte_device(&tensor));
        let less_than_zero = B::byte_lower_elem(tensor.clone(), 0.0f32.elem());
        let greater_than_zero = B::byte_greater_elem(tensor, 0.0f32.elem());

        let mut result = B::byte_mask_fill(zeros, less_than_zero, (-1.0f32).elem());
        result = B::byte_mask_fill(result, greater_than_zero, 1.0f32.elem());
        result
    }

    /// Broadcasts the int `tensor` to the given `shape`.
    fn byte_expand(tensor: ByteTensor<B>, shape: Shape) -> ByteTensor<B>;

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
    fn byte_sort(tensor: ByteTensor<B>, dim: usize, descending: bool) -> ByteTensor<B> {
        sort::<B, Byte>(tensor, dim, descending)
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
    fn byte_sort_with_indices(
        tensor: ByteTensor<B>,
        dim: usize,
        descending: bool,
    ) -> (ByteTensor<B>, IntTensor<B>) {
        sort_with_indices::<B, Byte>(tensor, dim, descending)
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
    fn byte_argsort(tensor: ByteTensor<B>, dim: usize, descending: bool) -> IntTensor<B> {
        argsort::<B, Byte>(tensor, dim, descending)
    }
}
