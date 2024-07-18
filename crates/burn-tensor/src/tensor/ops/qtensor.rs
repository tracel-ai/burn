use alloc::vec::Vec;
use core::{future::Future, ops::Range};

use crate::{
    backend::Backend,
    quantization::{QTensorPrimitive, QuantizationParametersPrimitive, QuantizationScheme},
    Device, Shape, TensorData,
};

use super::{BoolTensor, FloatElem, FloatTensor, IntElem, IntTensor, QuantizedTensor};

/// Quantized Tensor API for basic operations, see [tensor](crate::Tensor)
/// for documentation on each function.
pub trait QTensorOps<B: Backend> {
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
    fn q_from_data<const D: usize>(data: TensorData, device: &Device<B>) -> QuantizedTensor<B, D>;

    /// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
    fn quantize<const D: usize>(
        tensor: FloatTensor<B, D>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<B>,
    ) -> QuantizedTensor<B, D>;

    /// Dynamically convert the tensor to a lower precision data type based on the quantization scheme.
    fn quantize_dynamic<const D: usize>(
        tensor: FloatTensor<B, D>,
        scheme: &QuantizationScheme,
    ) -> QuantizedTensor<B, D> {
        // Dynamically compute min/max tensor range and qparams before quantizing
        let min = B::float_min(tensor.clone());
        let max = B::float_max(tensor.clone());
        let qparams = scheme.compute_q_params_primitive(min, max);
        Self::quantize(tensor, scheme, qparams)
    }

    /// Convert the tensor back to a higher precision data type.
    fn dequantize<const D: usize>(tensor: QuantizedTensor<B, D>) -> FloatTensor<B, D>;

    /// Gets the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn q_shape<const D: usize>(tensor: &QuantizedTensor<B, D>) -> Shape<D>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn q_device<const D: usize>(tensor: &QuantizedTensor<B, D>) -> Device<B>;

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
    fn q_to_device<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        device: &Device<B>,
    ) -> QuantizedTensor<B, D>;

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
    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<B, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<B, D2>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn q_into_data<const D: usize>(
        tensor: QuantizedTensor<B, D>,
    ) -> impl Future<Output = TensorData> + Send;

    /// Detaches a tensor from the computation graph.
    fn q_detach<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Sets the `require_grad` flag of a tensor.
    fn q_set_require_grad<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        _require_grad: bool,
    ) -> QuantizedTensor<B, D> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Returns the `require_grad` flag of a tensor.
    fn q_is_require_grad<const D: usize>(_tensor: &QuantizedTensor<B, D>) -> bool {
        // Should only be overridden by autodiff backends.
        false
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
    fn q_add<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        // Heuristic: prioritize lhs scheme
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);
        let out_f = B::float_add(lhs_f, rhs_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_add_scalar<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_add_scalar(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_clamp_min<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        min: FloatElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_clamp_min(tensor_f, min);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_clamp_max<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        max: FloatElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_clamp_max(tensor_f, max);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_clamp<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        min: FloatElem<B>,
        max: FloatElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_clamp(tensor_f, min, max);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_sub<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        // Heuristic: prioritize lhs scheme
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);
        let out_f = B::float_sub(lhs_f, rhs_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_sub_scalar<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_sub_scalar(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Multiplies two tensors together element-wise.
    fn q_mul<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        // Heuristic: prioritize lhs scheme
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);
        let out_f = B::float_mul(lhs_f, rhs_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_mul_scalar<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_mul_scalar(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_div<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        // Heuristic: prioritize lhs scheme
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);
        let out_f = B::float_div(lhs_f, rhs_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_div_scalar<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_div_scalar(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Computes the modulus of a tensor given a scalar.
    ///
    /// # Arguments
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The result of applying the modulus of the scalar to the tensor.
    fn q_remainder_scalar<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_remainder_scalar(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_matmul<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        // Heuristic: prioritize lhs scheme
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);
        let out_f = B::float_matmul(lhs_f, rhs_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Negates a tensor element-wise.
    fn q_neg<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_neg(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Calculates the reciprocals element-wise
    fn q_recip<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_recip(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_transpose<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        Self::q_swap_dims(tensor, D - 2, D - 1)
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
    fn q_swap_dims<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<B, D>;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn q_permute<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        axes: [usize; D],
    ) -> QuantizedTensor<B, D>;

    /// Reverse the order of elements in a tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reverse.
    /// * `axes` - The axes to reverse.
    ///
    /// The tensor with the elements reversed.
    fn q_flip<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        axes: &[usize],
    ) -> QuantizedTensor<B, D>;

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
    fn q_gather<const D: usize>(
        dim: usize,
        tensor: QuantizedTensor<B, D>,
        indices: IntTensor<B, D>,
    ) -> QuantizedTensor<B, D>;

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
    fn q_scatter<const D: usize>(
        dim: usize,
        tensor: QuantizedTensor<B, D>,
        indices: IntTensor<B, D>,
        value: QuantizedTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let value_f = Self::dequantize(value);
        let out_f = B::float_scatter(dim, tensor_f, indices, value_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_select<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
    ) -> QuantizedTensor<B, D>;

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
    fn q_select_assign<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
        indices: IntTensor<B, 1>,
        value: QuantizedTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let value_f = Self::dequantize(value);
        let out_f = B::float_select_assign(tensor_f, dim, indices, value_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_slice<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<B, D1>,
        ranges: [Range<usize>; D2],
    ) -> QuantizedTensor<B, D1>;

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
    fn q_slice_assign<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<B, D1>,
        ranges: [Range<usize>; D2],
        value: QuantizedTensor<B, D1>,
    ) -> QuantizedTensor<B, D1> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let value_f = Self::dequantize(value);
        let out_f = B::float_slice_assign(tensor_f, ranges, value_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_mask_where<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        mask: BoolTensor<B, D>,
        value: QuantizedTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let value_f = Self::dequantize(value);
        let out_f = B::float_mask_where(tensor_f, mask, value_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_mask_fill<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        mask: BoolTensor<B, D>,
        value: FloatElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_mask_fill(tensor_f, mask, value);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_equal<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);

        B::float_equal(lhs_f, rhs_f)
    }

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
    fn q_not_equal<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);

        B::float_not_equal(lhs_f, rhs_f)
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
    fn q_equal_elem<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);

        B::float_equal_elem(lhs_f, rhs)
    }

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
    fn q_not_equal_elem<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);

        B::float_not_equal_elem(lhs_f, rhs)
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
    fn q_greater<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);

        B::float_greater(lhs_f, rhs_f)
    }

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
    fn q_greater_elem<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);

        B::float_greater_elem(lhs_f, rhs)
    }

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
    fn q_greater_equal<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);

        B::float_greater_equal(lhs_f, rhs_f)
    }

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
    fn q_greater_equal_elem<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);

        B::float_greater_equal_elem(lhs_f, rhs)
    }

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
    fn q_lower<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);

        B::float_lower(lhs_f, rhs_f)
    }

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
    fn q_lower_elem<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);

        B::float_lower_elem(lhs_f, rhs)
    }

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
    fn q_lower_equal<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: QuantizedTensor<B, D>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);
        let rhs_f = Self::dequantize(rhs);

        B::float_lower_equal(lhs_f, rhs_f)
    }

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
    fn q_lower_equal_elem<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatElem<B>,
    ) -> BoolTensor<B, D> {
        let lhs_f = Self::dequantize(lhs);

        B::float_lower_equal_elem(lhs_f, rhs)
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
    fn q_sum<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, 1> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_sum(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_sum_dim<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_sum_dim(tensor_f, dim);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Product of all elements in a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to product.
    ///
    /// # Returns
    ///
    /// A scalar tensor with the product of all elements in `tensor`.
    fn q_prod<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, 1> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_prod(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_prod_dim<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_prod_dim(tensor_f, dim);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_mean<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, 1> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_mean(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_mean_dim<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_mean_dim(tensor_f, dim);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Returns a new tensor with exponential values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to exponentiate.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with exponential values.
    fn q_exp<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_exp(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Returns a new tensor with natural logarithm values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the logarithm of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with natural logarithm values.
    fn q_log<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_log(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Returns a new tensor with logarithm values of (1 + Xi).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the logarithm of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with logarithm values of (1 + Xi).
    fn q_log1p<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_log1p(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_powf<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: FloatTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_powf(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_powi<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: IntTensor<B, D>,
    ) -> QuantizedTensor<B, D> {
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_powi(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Element-wise power with an int scalar.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the value of `rhs`.
    fn q_powi_scalar<const D: usize>(
        lhs: QuantizedTensor<B, D>,
        rhs: IntElem<B>,
    ) -> QuantizedTensor<B, D> {
        let scheme = lhs.scheme().clone();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_powi_scalar(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Element-wise power with a float scalar.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to exponentiate.
    /// * `value` - The exponent.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with values raised to the power of `value`.
    fn q_powf_scalar<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        value: f32,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_powf_scalar(tensor_f, value);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Returns a new tensor with square root values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the square root of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with square root values.
    fn q_sqrt<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_sqrt(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_abs<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_abs(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Returns a new tensor with cosine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the cosine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with cosine values.
    fn q_cos<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_cos(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Returns a new tensor with sine values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the sine of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with sine values.
    fn q_sin<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_sin(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Returns a new tensor with tangent values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the tangent of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with tangent values.
    fn q_tanh<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_tanh(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Returns a new tensor with the error function values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to take the error function of.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as `tensor` with error function values.
    fn q_erf<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_erf(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_cat<const D: usize>(
        tensors: Vec<QuantizedTensor<B, D>>,
        dim: usize,
    ) -> QuantizedTensor<B, D> {
        // Heuristic: prioritize first tensor scheme
        let scheme = tensors.first().unwrap().scheme().clone();

        let tensor_f = tensors
            .into_iter()
            .map(|tensor| Self::dequantize(tensor))
            .collect();

        let out_f = B::float_cat(tensor_f, dim);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_argmax<const D: usize>(tensor: QuantizedTensor<B, D>, dim: usize) -> IntTensor<B, D>;

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
    fn q_argmin<const D: usize>(tensor: QuantizedTensor<B, D>, dim: usize) -> IntTensor<B, D>;

    /// Gets the maximum element of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    ///
    /// # Returns
    ///
    /// A tensor with the maximum element of `tensor`.
    fn q_max<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, 1> {
        let shape = B::q_shape(&tensor);
        let tensor = B::q_reshape(tensor, Shape::new([shape.num_elements()]));

        B::q_max_dim(tensor, 0)
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
    fn q_max_dim<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
    ) -> QuantizedTensor<B, D> {
        let index = B::q_argmax(tensor.clone(), dim);

        B::q_gather(dim, tensor, index)
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
    fn q_max_dim_with_indices<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
    ) -> (QuantizedTensor<B, D>, IntTensor<B, D>) {
        let index = B::q_argmax(tensor.clone(), dim);
        let values = B::q_gather(dim, tensor, index.clone());

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
    fn q_min<const D: usize>(tensor: QuantizedTensor<B, D>) -> QuantizedTensor<B, 1> {
        let shape = B::q_shape(&tensor);
        let tensor = B::q_reshape(tensor, Shape::new([shape.num_elements()]));

        B::q_min_dim(tensor, 0)
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
    fn q_min_dim<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
    ) -> QuantizedTensor<B, D> {
        let index = B::q_argmin(tensor.clone(), dim);

        B::q_gather(dim, tensor, index)
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
    fn q_min_dim_with_indices<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
    ) -> (QuantizedTensor<B, D>, IntTensor<B, D>) {
        let index = B::q_argmin(tensor.clone(), dim);
        let values = B::q_gather(dim, tensor, index.clone());

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
    fn q_narrow<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> QuantizedTensor<B, D> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_narrow(tensor_f, dim, start, length);

        Self::quantize_dynamic(out_f, &scheme)
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
    /// A vector of tensors
    fn q_chunk<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<QuantizedTensor<B, D>> {
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_chunk(tensor_f, chunks, dim);

        out_f
            .into_iter()
            .map(|tensor| Self::quantize_dynamic(tensor, &scheme))
            .collect()
    }

    /// Tests if any element in the `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if any element in the tensor is True, False otherwise.
    fn q_any<const D: usize>(tensor: QuantizedTensor<B, D>) -> BoolTensor<B, 1> {
        let tensor_f = Self::dequantize(tensor);
        B::float_any(tensor_f)
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
    fn q_any_dim<const D: usize>(tensor: QuantizedTensor<B, D>, dim: usize) -> BoolTensor<B, D> {
        let tensor_f = Self::dequantize(tensor);
        B::float_any_dim(tensor_f, dim)
    }

    /// Tests if all elements in the `tensor` evaluate to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, 1, Bool>` with a single element, True if all elements in the input tensor
    /// evaluate to True, False otherwise.
    fn q_all<const D: usize>(tensor: QuantizedTensor<B, D>) -> BoolTensor<B, 1> {
        let tensor_f = Self::dequantize(tensor);
        B::float_all(tensor_f)
    }

    /// Tests if all elements in the `tensor` evaluate to True along a given dimension `dim`.
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
    fn q_all_dim<const D: usize>(tensor: QuantizedTensor<B, D>, dim: usize) -> BoolTensor<B, D> {
        let tensor_f = Self::dequantize(tensor);
        B::float_all_dim(tensor_f, dim)
    }

    /// Broadcasts the `tensor` to the given `shape`.
    fn q_expand<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<B, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<B, D2>;

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
    fn q_sort<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
        descending: bool,
    ) -> QuantizedTensor<B, D> {
        // Default implementation. Backends can sort on the int values since qparams remain the same.
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_sort(tensor_f, dim, descending);

        Self::quantize_dynamic(out_f, &scheme)
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
    fn q_sort_with_indices<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
        descending: bool,
    ) -> (QuantizedTensor<B, D>, IntTensor<B, D>) {
        // Default implementation. Backends can sort on the int values since qparams remain the same.
        let scheme = tensor.scheme().clone();

        let tensor_f = Self::dequantize(tensor);
        let (out_f, indices) = B::float_sort_with_indices(tensor_f, dim, descending);

        (Self::quantize_dynamic(out_f, &scheme), indices)
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
    fn q_argsort<const D: usize>(
        tensor: QuantizedTensor<B, D>,
        dim: usize,
        descending: bool,
    ) -> IntTensor<B, D> {
        // Default implementation. Backends can sort on the int values since qparams remain the same.
        let tensor_f = Self::dequantize(tensor);
        B::float_argsort(tensor_f, dim, descending)
    }
}
