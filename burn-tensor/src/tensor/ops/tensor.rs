use super::{BoolTensor, Device, FloatElem, FloatTensor, FullPrecisionBackend, IntElem, IntTensor};
use crate::{backend::Backend, tensor::Shape, Data, Distribution, ElementConversion, Float};
use crate::{tensor::api::chunk, tensor::api::narrow};
use alloc::vec::Vec;
use burn_common::reader::Reader;
use core::ops::Range;
use num_traits::ToPrimitive;

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
    fn float_from_data<const D: usize>(
        data: Data<FloatElem<B>, D>,
        device: &Device<B>,
    ) -> FloatTensor<B, D>;

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
    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<B>,
    ) -> FloatTensor<B, D>;

    /// Creates a new tensor with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and zeros.
    fn float_zeros<const D: usize>(shape: Shape<D>, device: &Device<B>) -> FloatTensor<B, D> {
        Self::float_from_data(Data::zeros(shape), device)
    }

    /// Creates a new tensor with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the given shape and ones.
    fn float_ones<const D: usize>(shape: Shape<D>, device: &Device<B>) -> FloatTensor<B, D> {
        Self::float_from_data(Data::ones(shape), device)
    }

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
    fn float_full<const D: usize>(
        shape: Shape<D>,
        fill_value: FloatElem<B>,
        device: &Device<B>,
    ) -> FloatTensor<B, D> {
        Self::float_add_scalar(Self::float_zeros(shape, device), fill_value)
    }

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn float_shape<const D: usize>(tensor: &FloatTensor<B, D>) -> Shape<D>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn float_to_data<const D: usize>(tensor: &FloatTensor<B, D>) -> Reader<Data<FloatElem<B>, D>> {
        Self::float_into_data(tensor.clone())
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
    fn float_into_data<const D: usize>(tensor: FloatTensor<B, D>) -> Reader<Data<FloatElem<B>, D>>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn float_device<const D: usize>(tensor: &FloatTensor<B, D>) -> Device<B>;

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
    fn float_to_device<const D: usize>(
        tensor: FloatTensor<B, D>,
        device: &Device<B>,
    ) -> FloatTensor<B, D>;

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
    fn float_arange(range: Range<i64>, device: &Device<B>) -> IntTensor<B, 1> {
        Self::float_arange_step(range, 1, device)
    }

    /// Converts float tensor to int tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The int tensor with the same data as the float tensor.
    fn float_into_int<const D: usize>(tensor: FloatTensor<B, D>) -> IntTensor<B, D>;

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
    fn float_arange_step(range: Range<i64>, step: usize, device: &Device<B>) -> IntTensor<B, 1> {
        let value = range
            .step_by(step)
            .map(|i| i.elem())
            .collect::<Vec<IntElem<B>>>();
        let shape = Shape::new([value.len()]);
        let data = Data::new(value, shape);
        B::int_from_data(data, device)
    }

    /// Creates an empty tensor with the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The empty tensor with the given shape.
    fn float_empty<const D: usize>(shape: Shape<D>, device: &Device<B>) -> FloatTensor<B, D>;

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
    fn float_repeat<const D: usize>(
        tensor: FloatTensor<B, D>,
        dim: usize,
        times: usize,
    ) -> FloatTensor<B, D> {
        let mut shape = B::float_shape(&tensor);
        if shape.dims[dim] != 1 {
            panic!("Can only repeat dimension with dim=1");
        }
        shape.dims[dim] = times;

        let mut i = 0;
        let indices_select_all = [0; D].map(|_| {
            let start = 0;
            let end = shape.dims[i];
            i += 1;
            start..end
        });

        let mut tensor_output = B::float_empty(shape, &B::float_device(&tensor));
        for i in 0..times {
            let mut indices = indices_select_all.clone();
            indices[dim] = i..i + 1;
            tensor_output = B::float_slice_assign(tensor_output, indices, tensor.clone());
        }

        tensor_output
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
    fn float_add<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_add_scalar<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> FloatTensor<B, D>;

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
    fn float_clamp_min<const D: usize>(
        tensor: FloatTensor<B, D>,
        min: FloatElem<B>,
    ) -> FloatTensor<B, D> {
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
    fn float_clamp_max<const D: usize>(
        tensor: FloatTensor<B, D>,
        max: FloatElem<B>,
    ) -> FloatTensor<B, D> {
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
    fn float_clamp<const D: usize>(
        tensor: FloatTensor<B, D>,
        min: FloatElem<B>,
        max: FloatElem<B>,
    ) -> FloatTensor<B, D> {
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
    fn float_sub<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_sub_scalar<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> FloatTensor<B, D>;

    /// Multiplies two tensors together element-wise.
    fn float_mul<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_mul_scalar<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> FloatTensor<B, D>;

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
    fn float_div<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_div_scalar<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> FloatTensor<B, D>;

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
    fn float_matmul<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

    /// Negates a tensor element-wise.
    fn float_neg<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D> {
        Self::float_mul_scalar(tensor, (-1.0_f32).elem::<FloatElem<B>>())
    }

    /// Calculates the reciprocals elementwise
    fn float_recip<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Transposes a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn float_transpose<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D> {
        Self::float_swap_dims(tensor, D - 2, D - 1)
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
    fn float_swap_dims<const D: usize>(
        tensor: FloatTensor<B, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<B, D>;

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
    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<B, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<B, D2>;

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
    fn float_gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<B, D>,
        indices: IntTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<B, D>,
        indices: IntTensor<B, D>,
        value: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_select<const D: usize>(
        tensor: FloatTensor<B, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
    ) -> FloatTensor<B, D>;

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
    fn float_select_assign<const D: usize>(
        tensor: FloatTensor<B, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
        value: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

    /// Select tensor elements corresponding for the given ranges.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `ranges` - The ranges to select.
    ///
    /// # Returns
    ///
    /// The selected elements in a new tensor.
    fn float_slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<B, D1>,
        ranges: [Range<usize>; D2],
    ) -> FloatTensor<B, D1>;

    /// Assign the selected elements corresponding for the given ranges to the given value.
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
    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<B, D1>,
        ranges: [Range<usize>; D2],
        value: FloatTensor<B, D1>,
    ) -> FloatTensor<B, D1>;

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
    fn float_mask_where<const D: usize>(
        tensor: FloatTensor<B, D>,
        mask: BoolTensor<B, D>,
        value: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

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
    fn float_mask_fill<const D: usize>(
        tensor: FloatTensor<B, D>,
        mask: BoolTensor<B, D>,
        value: FloatElem<B>,
    ) -> FloatTensor<B, D>;

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
    fn float_equal<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> BoolTensor<B, D>;

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
    fn float_equal_elem<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D>;

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
    fn float_greater<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> BoolTensor<B, D>;

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
    fn float_greater_elem<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D>;

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
    fn float_greater_equal<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> BoolTensor<B, D>;

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
    fn float_greater_equal_elem<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D>;

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
    fn float_lower<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> BoolTensor<B, D>;

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
    fn float_lower_elem<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D>;

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
    fn float_lower_equal<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> BoolTensor<B, D>;

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
    fn float_lower_equal_elem<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D>;

    /// Detaches a tensor from the computation graph.
    fn float_detach<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Sets the `require_grad` flag of a tensor.
    fn float_set_require_grad<const D: usize>(
        tensor: FloatTensor<B, D>,
        _require_grad: bool,
    ) -> FloatTensor<B, D> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Returns the `require_grad` flag of a tensor.
    fn float_is_require_grad<const D: usize>(_tensor: &FloatTensor<B, D>) -> bool {
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
    fn float_sum<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, 1>;

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
    fn float_sum_dim<const D: usize>(tensor: FloatTensor<B, D>, dim: usize) -> FloatTensor<B, D>;

    /// Mean of all elements in a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to mean.
    ///
    /// # Returns
    ///
    /// A scalar tensor with the mean of all elements in `tensor`.
    fn float_mean<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, 1> {
        let num_elems = B::float_shape(&tensor).num_elements();
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
    fn float_mean_dim<const D: usize>(tensor: FloatTensor<B, D>, dim: usize) -> FloatTensor<B, D>;

    /// Converts a tensor to full precision.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to convert.
    ///
    /// # Returns
    ///
    /// A tensor with the same values as `tensor` but with full precision.
    fn float_to_full_precision<const D: usize>(
        tensor: &FloatTensor<B, D>,
    ) -> FloatTensor<FullPrecisionBackend<B>, D>;

    /// Converts a tensor from full precision.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to convert.
    ///
    /// # Returns
    ///
    /// A tensor with the same values as `tensor` but with the precision of the backend.
    fn float_from_full_precision<const D: usize>(
        tensor: FloatTensor<FullPrecisionBackend<B>, D>,
    ) -> FloatTensor<B, D>;

    /// Returns a new tensor with exponential values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to exponentiate.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with exponential values.
    fn float_exp<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Returns a new tensor with natural logarithm values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the logarithm of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with natural logarithm values.
    fn float_log<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Returns a new tensor with logarithm values of (1 + Xi).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the logarithm of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with logarithm values of (1 + Xi).
    fn float_log1p<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Elementwise power with a FloatTensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of the elements of `rhs`.
    fn float_powf<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> FloatTensor<B, D>;

    /// Elementwise power with an IntTensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side floatTensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the value of `rhs`. Result is an IntTensor.
    fn float_powi<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: IntTensor<B, D>,
    ) -> FloatTensor<B, D> {
        Self::float_powf(lhs, B::int_into_float::<D>(rhs))
    }

    /// raises a tensor to the power of a int scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the value of `rhs`.
    fn float_powi_scalar<const D: usize>(
        lhs: FloatTensor<B, D>,
        rhs: IntElem<B>,
    ) -> FloatTensor<B, D> {
        Self::float_powf_scalar(lhs, rhs.to_f32().unwrap())
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
    fn float_powf_scalar<const D: usize>(
        tensor: FloatTensor<B, D>,
        value: f32,
    ) -> FloatTensor<B, D>;

    /// Returns a new tensor with square root values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the square root of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with square root values.
    fn float_sqrt<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Returns a new tensor with absolute values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take absolute value of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with absolute values.
    fn float_abs<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Returns a new tensor with cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the cosine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with cosine values.
    fn float_cos<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Returns a new tensor with sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the sine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with sine values.
    fn float_sin<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Returns a new tensor with tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the tangent of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with tangent values.
    fn float_tanh<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Returns a new tensor with the error function values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the error function of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with error function values.
    fn float_erf<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, D>;

    /// Catcatenates tensors along a dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors to catcatenate.
    /// * `dim` - The dimension along which to catcatenate.
    ///
    /// # Returns
    ///
    /// A tensor with the catcatenated tensors along `dim`.
    fn float_cat<const D: usize>(tensors: Vec<FloatTensor<B, D>>, dim: usize) -> FloatTensor<B, D>;

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
    fn float_argmax<const D: usize>(tensor: FloatTensor<B, D>, dim: usize) -> IntTensor<B, D>;

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
    fn float_argmin<const D: usize>(tensor: FloatTensor<B, D>, dim: usize) -> IntTensor<B, D>;

    /// Gets the maximum element of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    ///
    /// # Returns
    ///
    /// A tensor with the maximum element of `tensor`.
    fn float_max<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, 1> {
        let shape = B::float_shape(&tensor);
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
    fn float_max_dim<const D: usize>(tensor: FloatTensor<B, D>, dim: usize) -> FloatTensor<B, D> {
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
    fn float_max_dim_with_indices<const D: usize>(
        tensor: FloatTensor<B, D>,
        dim: usize,
    ) -> (FloatTensor<B, D>, IntTensor<B, D>) {
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
    fn float_min<const D: usize>(tensor: FloatTensor<B, D>) -> FloatTensor<B, 1> {
        let shape = B::float_shape(&tensor);
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
    fn float_min_dim<const D: usize>(tensor: FloatTensor<B, D>, dim: usize) -> FloatTensor<B, D> {
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
    fn float_min_dim_with_indices<const D: usize>(
        tensor: FloatTensor<B, D>,
        dim: usize,
    ) -> (FloatTensor<B, D>, IntTensor<B, D>) {
        let index = B::float_argmin(tensor.clone(), dim);
        let values = B::float_gather(dim, tensor, index.clone());

        (values, index)
    }

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
    fn float_narrow<const D: usize>(
        tensor: FloatTensor<B, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> FloatTensor<B, D> {
        narrow::<B, D, Float>(tensor, dim, start, length)
    }

    /// Split the tensor along the given dimension into chunks.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `chunks` - The number of chunks to be produced
    /// * `times` - The dimension along which the tensor will be split.
    ///
    /// # Returns
    ///
    /// A vectors of tensors
    ///
    fn float_chunk<const D: usize>(
        tensor: FloatTensor<B, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<FloatTensor<B, D>> {
        chunk::<B, D, Float>(tensor, chunks, dim)
    }
}
