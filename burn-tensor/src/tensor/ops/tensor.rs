use alloc::vec::Vec;
use core::ops::Range;

use crate::{backend::Backend, tensor::Shape, Data, Distribution, ElementConversion};

/// Operations on float tensors.
pub trait TensorOps<B: Backend> {
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
    fn from_data<const D: usize>(
        data: Data<B::FloatElem, D>,
        device: &B::Device,
    ) -> B::TensorPrimitive<D>;

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
    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<B::FloatElem>,
        device: &B::Device,
    ) -> B::TensorPrimitive<D>;

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
    fn zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> B::TensorPrimitive<D> {
        Self::from_data(Data::zeros(shape), device)
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
    fn ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> B::TensorPrimitive<D> {
        Self::from_data(Data::ones(shape), device)
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
    fn full<const D: usize>(
        shape: Shape<D>,
        fill_value: B::FloatElem,
        device: &B::Device,
    ) -> B::TensorPrimitive<D> {
        Self::add_scalar(Self::zeros(shape, device), fill_value)
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
    fn shape<const D: usize>(tensor: &B::TensorPrimitive<D>) -> Shape<D>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn to_data<const D: usize>(tensor: &B::TensorPrimitive<D>) -> Data<B::FloatElem, D>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn into_data<const D: usize>(tensor: B::TensorPrimitive<D>) -> Data<B::FloatElem, D> {
        Self::to_data(&tensor)
    }

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn device<const D: usize>(tensor: &B::TensorPrimitive<D>) -> B::Device;

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
    fn to_device<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        device: &B::Device,
    ) -> B::TensorPrimitive<D>;

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
    fn arange(range: Range<usize>, device: &B::Device) -> B::IntTensorPrimitive<1> {
        Self::arange_step(range, 1, device)
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
    fn into_int<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::IntTensorPrimitive<D>;

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
    fn arange_step(
        range: Range<usize>,
        step: usize,
        device: &B::Device,
    ) -> B::IntTensorPrimitive<1> {
        let value = range
            .step_by(step)
            .map(|i| (i as i64).elem())
            .collect::<Vec<B::IntElem>>();
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
    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> B::TensorPrimitive<D>;

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
    fn repeat<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        dim: usize,
        times: usize,
    ) -> B::TensorPrimitive<D> {
        let mut shape = B::shape(&tensor);
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

        let mut tensor_output = B::empty(shape, &B::device(&tensor));
        for i in 0..times {
            let mut indices = indices_select_all.clone();
            indices[dim] = i..i + 1;
            tensor_output = B::slice_assign(tensor_output, indices, tensor.clone());
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
    fn add<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

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
    fn add_scalar<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::FloatElem,
    ) -> B::TensorPrimitive<D>;

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
    fn clamp_min<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        min: B::FloatElem,
    ) -> B::TensorPrimitive<D> {
        // Default implementation
        let mask = Self::lower_elem(tensor.clone(), min);
        B::mask_fill(tensor, mask, min)
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
    fn clamp_max<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        max: B::FloatElem,
    ) -> B::TensorPrimitive<D> {
        // Default implementation
        let mask = Self::greater_elem(tensor.clone(), max);
        B::mask_fill(tensor, mask, max)
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
    fn clamp<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        min: B::FloatElem,
        max: B::FloatElem,
    ) -> B::TensorPrimitive<D> {
        // Default implementation
        Self::clamp_min(Self::clamp_max(tensor, max), min)
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
    fn sub<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

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
    fn sub_scalar<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::FloatElem,
    ) -> B::TensorPrimitive<D>;

    /// Multiplies two tensors together element-wise.
    fn mul<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

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
    fn mul_scalar<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::FloatElem,
    ) -> B::TensorPrimitive<D>;

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
    fn div<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

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
    fn div_scalar<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::FloatElem,
    ) -> B::TensorPrimitive<D>;

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
    fn matmul<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

    /// Negates a tensor element-wise.
    fn neg<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D> {
        Self::mul_scalar(tensor, (-1.0_f32).elem::<B::FloatElem>())
    }

    /// Transposes a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn transpose<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D> {
        Self::swap_dims(tensor, D - 2, D - 1)
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
    fn swap_dims<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> B::TensorPrimitive<D>;

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
    fn reshape<const D1: usize, const D2: usize>(
        tensor: B::TensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> B::TensorPrimitive<D2>;

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
    fn gather<const D: usize>(
        dim: usize,
        tensor: B::TensorPrimitive<D>,
        indices: B::IntTensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

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
    fn scatter<const D: usize>(
        dim: usize,
        tensor: B::TensorPrimitive<D>,
        indices: B::IntTensorPrimitive<D>,
        value: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

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
    fn select<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        dim: usize,
        indices: B::IntTensorPrimitive<1>,
    ) -> B::TensorPrimitive<D>;

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
    fn select_assign<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        dim: usize,
        indices: B::IntTensorPrimitive<1>,
        value: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

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
    fn slice<const D1: usize, const D2: usize>(
        tensor: B::TensorPrimitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> B::TensorPrimitive<D1>;

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
    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: B::TensorPrimitive<D1>,
        ranges: [Range<usize>; D2],
        value: B::TensorPrimitive<D1>,
    ) -> B::TensorPrimitive<D1>;

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
    fn mask_where<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        mask: B::BoolTensorPrimitive<D>,
        value: B::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

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
    fn mask_fill<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        mask: B::BoolTensorPrimitive<D>,
        value: B::FloatElem,
    ) -> B::TensorPrimitive<D>;

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
    fn equal<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;

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
    fn equal_elem<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::FloatElem,
    ) -> B::BoolTensorPrimitive<D>;

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
    fn greater<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;

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
    fn greater_elem<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::FloatElem,
    ) -> B::BoolTensorPrimitive<D>;

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
    fn greater_equal<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;

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
    fn greater_equal_elem<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::FloatElem,
    ) -> B::BoolTensorPrimitive<D>;

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
    fn lower<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;

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
    fn lower_elem<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::FloatElem,
    ) -> B::BoolTensorPrimitive<D>;

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
    fn lower_equal<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::TensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;

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
    fn lower_equal_elem<const D: usize>(
        lhs: B::TensorPrimitive<D>,
        rhs: B::FloatElem,
    ) -> B::BoolTensorPrimitive<D>;

    /// Detaches a tensor from the computation graph.
    fn detach<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Sets the `require_grad` flag of a tensor.
    fn set_require_grad<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        _require_grad: bool,
    ) -> B::TensorPrimitive<D> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Returns the `require_grad` flag of a tensor.
    fn is_require_grad<const D: usize>(_tensor: &B::TensorPrimitive<D>) -> bool {
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
    fn sum<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<1>;

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
    fn sum_dim<const D: usize>(tensor: B::TensorPrimitive<D>, dim: usize) -> B::TensorPrimitive<D>;

    /// Mean of all elements in a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to mean.
    ///
    /// # Returns
    ///
    /// A scalar tensor with the mean of all elements in `tensor`.
    fn mean<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<1> {
        let num_elems = B::shape(&tensor).num_elements();
        B::div_scalar(B::sum(tensor), (num_elems as i64).elem())
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
    fn mean_dim<const D: usize>(tensor: B::TensorPrimitive<D>, dim: usize)
        -> B::TensorPrimitive<D>;

    /// Converts a tensor to full precision.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to convert.
    ///
    /// # Returns
    ///
    /// A tensor with the same values as `tensor` but with full precision.
    fn to_full_precision<const D: usize>(
        tensor: &B::TensorPrimitive<D>,
    ) -> <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>;

    /// Converts a tensor from full precision.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to convert.
    ///
    /// # Returns
    ///
    /// A tensor with the same values as `tensor` but with the precision of the backend.
    fn from_full_precision<const D: usize>(
        tensor: <B::FullPrecisionBackend as Backend>::TensorPrimitive<D>,
    ) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with exponential values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to exponentiate.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with exponential values.
    fn exp<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with natural logarithm values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the logarithm of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with natural logarithm values.
    fn log<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with logarithm values of (1 + Xi).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the logarithm of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with logarithm values of (1 + Xi).
    fn log1p<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with values raised to the power of `value`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to exponentiate.
    /// * `value` - The exponent.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with values raised to the power of `value`.
    fn powf<const D: usize>(tensor: B::TensorPrimitive<D>, value: f32) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with square root values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the square root of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with square root values.
    fn sqrt<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with absolute values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take absolute value of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with absolute values.
    fn abs<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the cosine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with cosine values.
    fn cos<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the sine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with sine values.
    fn sin<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the tangent of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with tangent values.
    fn tanh<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;

    /// Returns a new tensor with the error function values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the error function of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with error function values.
    fn erf<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<D>;

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
    fn cat<const D: usize>(
        tensors: Vec<B::TensorPrimitive<D>>,
        dim: usize,
    ) -> B::TensorPrimitive<D>;

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
    fn argmax<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        dim: usize,
    ) -> B::IntTensorPrimitive<D>;

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
    fn argmin<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        dim: usize,
    ) -> B::IntTensorPrimitive<D>;

    /// Gets the maximum element of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    ///
    /// # Returns
    ///
    /// A tensor with the maximum element of `tensor`.
    fn max<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<1> {
        let shape = B::shape(&tensor);
        let tensor = B::reshape(tensor, Shape::new([shape.num_elements()]));

        B::max_dim(tensor, 0)
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
    fn max_dim<const D: usize>(tensor: B::TensorPrimitive<D>, dim: usize) -> B::TensorPrimitive<D> {
        let index = B::argmax(tensor.clone(), dim);

        B::gather(D - 1, tensor, index)
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
    fn max_dim_with_indices<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        dim: usize,
    ) -> (B::TensorPrimitive<D>, B::IntTensorPrimitive<D>) {
        let index = B::argmax(tensor.clone(), dim);
        let values = B::gather(D - 1, tensor, index.clone());

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
    fn min<const D: usize>(tensor: B::TensorPrimitive<D>) -> B::TensorPrimitive<1> {
        let shape = B::shape(&tensor);
        let tensor = B::reshape(tensor, Shape::new([shape.num_elements()]));

        B::min_dim(tensor, 0)
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
    fn min_dim<const D: usize>(tensor: B::TensorPrimitive<D>, dim: usize) -> B::TensorPrimitive<D> {
        let index = B::argmin(tensor.clone(), dim);

        B::gather(D - 1, tensor, index)
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
    fn min_dim_with_indices<const D: usize>(
        tensor: B::TensorPrimitive<D>,
        dim: usize,
    ) -> (B::TensorPrimitive<D>, B::IntTensorPrimitive<D>) {
        let index = B::argmin(tensor.clone(), dim);
        let values = B::gather(D - 1, tensor, index.clone());

        (values, index)
    }
}
