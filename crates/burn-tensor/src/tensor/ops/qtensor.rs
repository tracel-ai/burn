use alloc::vec::Vec;
use core::{future::Future, ops::Range};

use crate::{
    backend::Backend,
    quantization::{
        Calibration, QTensorPrimitive, QuantizationParametersPrimitive, QuantizationScheme,
    },
    Device, Shape, TensorData, TensorMetadata,
};

use super::{BoolTensor, FloatElem, FloatTensor, IntElem, IntTensor, QuantizedTensor};

/// Automatically applies dequantization -> float operation -> quantization.
#[macro_export]
macro_rules! dequant_op_quant {
    // Binary tensor float op w/ lhs & rhs
    (
        ty $ty:ty, float_op $float_op:expr, $t1:expr, $t2:expr
    ) => {{
        // Heuristic: prioritize lhs scheme
        let scheme = $t1.scheme().clone();

        let t1_f = <$ty>::dequantize($t1);
        let t2_f = <$ty>::dequantize($t2);
        #[allow(clippy::redundant_closure_call)]
        let out_f = $float_op(t1_f, t2_f);

        <$ty>::quantize_dynamic(out_f, &scheme)
    }};
    // Unary tensor float op
    (
        ty $ty:ty, float_op $float_op:expr, $tensor:expr
    ) => {{
        let scheme = $tensor.scheme().clone();

        let tensor_f = <$ty>::dequantize($tensor);
        #[allow(clippy::redundant_closure_call)]
        let out_f = $float_op(tensor_f);

        <$ty>::quantize_dynamic(out_f, &scheme)
    }};
}

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
    fn q_from_data(data: TensorData, device: &Device<B>) -> QuantizedTensor<B>;

    /// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
    fn quantize(
        tensor: FloatTensor<B>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<B>,
    ) -> QuantizedTensor<B>;

    /// Dynamically convert the tensor to a lower precision data type based on the quantization scheme.
    fn quantize_dynamic(tensor: FloatTensor<B>, scheme: &QuantizationScheme) -> QuantizedTensor<B> {
        // Dynamically compute min/max tensor range and qparams before quantizing
        let (min, max) = scheme.compute_range_primitive::<B>(tensor.clone(), &Calibration::MinMax);
        let qparams = scheme.compute_q_params_primitive(min, max);
        Self::quantize(tensor, scheme, qparams)
    }

    /// Convert the tensor back to a higher precision data type.
    fn dequantize(tensor: QuantizedTensor<B>) -> FloatTensor<B>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn q_device(tensor: &QuantizedTensor<B>) -> Device<B>;

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
    fn q_to_device(tensor: QuantizedTensor<B>, device: &Device<B>) -> QuantizedTensor<B>;

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
    fn q_reshape(tensor: QuantizedTensor<B>, shape: Shape) -> QuantizedTensor<B>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn q_into_data(tensor: QuantizedTensor<B>)
        -> impl Future<Output = TensorData> + 'static + Send;

    /// Detaches a tensor from the computation graph.
    fn q_detach(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Sets the `require_grad` flag of a tensor.
    fn q_set_require_grad(tensor: QuantizedTensor<B>, _require_grad: bool) -> QuantizedTensor<B> {
        // Should only be overridden by autodiff backends.
        tensor
    }

    /// Returns the `require_grad` flag of a tensor.
    fn q_is_require_grad(_tensor: &QuantizedTensor<B>) -> bool {
        // Should only be overridden by autodiff backends.
        false
    }

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
    fn q_repeat_dim(tensor: QuantizedTensor<B>, dim: usize, times: usize) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_repeat_dim(tensor, dim, times),
            tensor
        )
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
    fn q_add(lhs: QuantizedTensor<B>, rhs: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |lhs, rhs| B::float_add(lhs, rhs),
            lhs,
            rhs
        )
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
    fn q_add_scalar(lhs: QuantizedTensor<B>, rhs: FloatElem<B>) -> QuantizedTensor<B> {
        let scheme = *lhs.scheme();

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
    fn q_clamp_min(tensor: QuantizedTensor<B>, min: FloatElem<B>) -> QuantizedTensor<B> {
        let scheme = *tensor.scheme();

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
    fn q_clamp_max(tensor: QuantizedTensor<B>, max: FloatElem<B>) -> QuantizedTensor<B> {
        let scheme = *tensor.scheme();

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
    fn q_clamp(
        tensor: QuantizedTensor<B>,
        min: FloatElem<B>,
        max: FloatElem<B>,
    ) -> QuantizedTensor<B> {
        let scheme = *tensor.scheme();

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
    fn q_sub(lhs: QuantizedTensor<B>, rhs: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |lhs, rhs| B::float_sub(lhs, rhs),
            lhs,
            rhs
        )
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
    fn q_sub_scalar(lhs: QuantizedTensor<B>, rhs: FloatElem<B>) -> QuantizedTensor<B> {
        let scheme = *lhs.scheme();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_sub_scalar(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Multiplies two tensors together element-wise.
    fn q_mul(lhs: QuantizedTensor<B>, rhs: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |lhs, rhs| B::float_mul(lhs, rhs),
            lhs,
            rhs
        )
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
    fn q_mul_scalar(lhs: QuantizedTensor<B>, rhs: FloatElem<B>) -> QuantizedTensor<B> {
        let scheme = *lhs.scheme();

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
    fn q_div(lhs: QuantizedTensor<B>, rhs: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |lhs, rhs| B::float_div(lhs, rhs),
            lhs,
            rhs
        )
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
    fn q_div_scalar(lhs: QuantizedTensor<B>, rhs: FloatElem<B>) -> QuantizedTensor<B> {
        let scheme = *lhs.scheme();

        let lhs_f = Self::dequantize(lhs);
        let out_f = B::float_div_scalar(lhs_f, rhs);

        Self::quantize_dynamic(out_f, &scheme)
    }

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
    fn q_remainder(lhs: QuantizedTensor<B>, rhs: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |lhs, rhs| B::float_remainder(lhs, rhs),
            lhs,
            rhs
        )
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
    fn q_remainder_scalar(lhs: QuantizedTensor<B>, rhs: FloatElem<B>) -> QuantizedTensor<B> {
        let scheme = *lhs.scheme();

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
    fn q_matmul(lhs: QuantizedTensor<B>, rhs: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |lhs, rhs| B::float_matmul(lhs, rhs),
            lhs,
            rhs
        )
    }

    /// Negates a tensor element-wise.
    fn q_neg(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        let scheme = *tensor.scheme();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_neg(tensor_f);

        Self::quantize_dynamic(out_f, &scheme)
    }

    /// Calculates the reciprocals element-wise
    fn q_recip(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        let scheme = *tensor.scheme();

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
    fn q_transpose(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        let ndims = tensor.shape().num_dims();
        Self::q_swap_dims(tensor, ndims - 2, ndims - 1)
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
    fn q_swap_dims(tensor: QuantizedTensor<B>, dim1: usize, dim2: usize) -> QuantizedTensor<B>;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn q_permute(tensor: QuantizedTensor<B>, axes: &[usize]) -> QuantizedTensor<B>;

    /// Reverse the order of elements in a tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reverse.
    /// * `axes` - The axes to reverse.
    ///
    /// The tensor with the elements reversed.
    fn q_flip(tensor: QuantizedTensor<B>, axes: &[usize]) -> QuantizedTensor<B>;

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
    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<B>,
        indices: IntTensor<B>,
    ) -> QuantizedTensor<B> {
        // Default implementation. Backends can gather on the quantized values when supported.
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_gather(dim, tensor, indices),
            tensor
        )
    }

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
    fn q_scatter(
        dim: usize,
        tensor: QuantizedTensor<B>,
        indices: IntTensor<B>,
        value: QuantizedTensor<B>,
    ) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor, value| B::float_scatter(dim, tensor, indices, value),
            tensor,
            value
        )
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
    fn q_select(
        tensor: QuantizedTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
    ) -> QuantizedTensor<B>;

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
    fn q_select_assign(
        tensor: QuantizedTensor<B>,
        dim: usize,
        indices: IntTensor<B>,
        value: QuantizedTensor<B>,
    ) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor, value| B::float_select_assign(tensor, dim, indices, value),
            tensor,
            value
        )
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
    fn q_slice(tensor: QuantizedTensor<B>, ranges: &[Range<usize>]) -> QuantizedTensor<B>;

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
    fn q_slice_assign(
        tensor: QuantizedTensor<B>,
        ranges: &[Range<usize>],
        value: QuantizedTensor<B>,
    ) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor, value| B::float_slice_assign(tensor, ranges, value),
            tensor,
            value
        )
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
    fn q_mask_where(
        tensor: QuantizedTensor<B>,
        mask: BoolTensor<B>,
        value: QuantizedTensor<B>,
    ) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor, value| B::float_mask_where(tensor, mask, value),
            tensor,
            value
        )
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
    fn q_mask_fill(
        tensor: QuantizedTensor<B>,
        mask: BoolTensor<B>,
        value: FloatElem<B>,
    ) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_mask_fill(tensor, mask, value),
            tensor
        )
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
    fn q_sum(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_sum(tensor),
            tensor
        )
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
    fn q_sum_dim(tensor: QuantizedTensor<B>, dim: usize) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_sum_dim(tensor, dim),
            tensor
        )
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
    fn q_prod(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_prod(tensor),
            tensor
        )
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
    fn q_prod_dim(tensor: QuantizedTensor<B>, dim: usize) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_prod_dim(tensor, dim),
            tensor
        )
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
    fn q_mean(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_mean(tensor),
            tensor
        )
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
    fn q_mean_dim(tensor: QuantizedTensor<B>, dim: usize) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_mean_dim(tensor, dim),
            tensor
        )
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
    fn q_exp(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_exp(tensor),
            tensor
        )
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
    fn q_log(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_log(tensor),
            tensor
        )
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
    fn q_log1p(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_log1p(tensor),
            tensor
        )
    }

    /// Element-wise power with another tensor.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The elements of `lhs` raised to the power of the elements of `rhs`.
    fn q_powf(lhs: QuantizedTensor<B>, rhs: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |lhs, rhs| B::float_powf(lhs, rhs),
            lhs,
            rhs
        )
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
    fn q_powi(lhs: QuantizedTensor<B>, rhs: IntTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_powi(tensor, rhs),
            lhs
        )
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
    fn q_powi_scalar(lhs: QuantizedTensor<B>, rhs: IntElem<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_powi_scalar(tensor, rhs),
            lhs
        )
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
    fn q_powf_scalar(tensor: QuantizedTensor<B>, value: f32) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_powf_scalar(tensor, value),
            tensor
        )
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
    fn q_sqrt(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_sqrt(tensor),
            tensor
        )
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
    fn q_abs(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_abs(tensor),
            tensor
        )
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
    fn q_cos(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_cos(tensor),
            tensor
        )
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
    fn q_sin(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_sin(tensor),
            tensor
        )
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
    fn q_tanh(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_tanh(tensor),
            tensor
        )
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
    fn q_erf(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_erf(tensor),
            tensor
        )
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
    fn q_cat(tensors: Vec<QuantizedTensor<B>>, dim: usize) -> QuantizedTensor<B> {
        // Heuristic: prioritize first tensor scheme
        let scheme = *tensors.first().unwrap().scheme();

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
    fn q_argmax(tensor: QuantizedTensor<B>, dim: usize) -> IntTensor<B> {
        // Default implementation. Backends can sort on the int values since qparams remain the same.
        let tensor_f = Self::dequantize(tensor);
        B::float_argmax(tensor_f, dim)
    }

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
    fn q_argmin(tensor: QuantizedTensor<B>, dim: usize) -> IntTensor<B> {
        // Default implementation. Backends can sort on the int values since qparams remain the same.
        let tensor_f = Self::dequantize(tensor);
        B::float_argmin(tensor_f, dim)
    }

    /// Gets the maximum element of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to get the maximum elements of.
    ///
    /// # Returns
    ///
    /// A tensor with the maximum element of `tensor`.
    fn q_max(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        let shape = tensor.shape();
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
    fn q_max_dim(tensor: QuantizedTensor<B>, dim: usize) -> QuantizedTensor<B> {
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
    fn q_max_dim_with_indices(
        tensor: QuantizedTensor<B>,
        dim: usize,
    ) -> (QuantizedTensor<B>, IntTensor<B>) {
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
    fn q_min(tensor: QuantizedTensor<B>) -> QuantizedTensor<B> {
        let shape = tensor.shape();
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
    fn q_min_dim(tensor: QuantizedTensor<B>, dim: usize) -> QuantizedTensor<B> {
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
    fn q_min_dim_with_indices(
        tensor: QuantizedTensor<B>,
        dim: usize,
    ) -> (QuantizedTensor<B>, IntTensor<B>) {
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
    fn q_narrow(
        tensor: QuantizedTensor<B>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> QuantizedTensor<B> {
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_narrow(tensor, dim, start, length),
            tensor
        )
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
    /// A vector of tensors.
    fn q_chunk(tensor: QuantizedTensor<B>, chunks: usize, dim: usize) -> Vec<QuantizedTensor<B>> {
        let scheme = *tensor.scheme();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_chunk(tensor_f, chunks, dim);

        out_f
            .into_iter()
            .map(|tensor| Self::quantize_dynamic(tensor, &scheme))
            .collect()
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
    fn q_split(
        tensor: QuantizedTensor<B>,
        split_size: usize,
        dim: usize,
    ) -> Vec<QuantizedTensor<B>> {
        let scheme = *tensor.scheme();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_split(tensor_f, split_size, dim);

        out_f
            .into_iter()
            .map(|tensor| Self::quantize_dynamic(tensor, &scheme))
            .collect()
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
    fn q_split_with_sizes(
        tensor: QuantizedTensor<B>,
        split_sizes: Vec<usize>,
        dim: usize,
    ) -> Vec<QuantizedTensor<B>> {
        let scheme = *tensor.scheme();

        let tensor_f = Self::dequantize(tensor);
        let out_f = B::float_split_with_sizes(tensor_f, split_sizes, dim);

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
    fn q_any(tensor: QuantizedTensor<B>) -> BoolTensor<B> {
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
    fn q_any_dim(tensor: QuantizedTensor<B>, dim: usize) -> BoolTensor<B> {
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
    fn q_all(tensor: QuantizedTensor<B>) -> BoolTensor<B> {
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
    fn q_all_dim(tensor: QuantizedTensor<B>, dim: usize) -> BoolTensor<B> {
        let tensor_f = Self::dequantize(tensor);
        B::float_all_dim(tensor_f, dim)
    }

    /// Broadcasts the `tensor` to the given `shape`.
    fn q_expand(tensor: QuantizedTensor<B>, shape: Shape) -> QuantizedTensor<B>;

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
    fn q_sort(tensor: QuantizedTensor<B>, dim: usize, descending: bool) -> QuantizedTensor<B> {
        // Default implementation. Backends can sort on the int values since qparams remain the same.
        dequant_op_quant!(
            ty Self,
            float_op |tensor| B::float_sort(tensor, dim, descending),
            tensor
        )
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
    fn q_sort_with_indices(
        tensor: QuantizedTensor<B>,
        dim: usize,
        descending: bool,
    ) -> (QuantizedTensor<B>, IntTensor<B>) {
        // Default implementation. Backends can sort on the int values since qparams remain the same.
        let scheme = *tensor.scheme();

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
    fn q_argsort(tensor: QuantizedTensor<B>, dim: usize, descending: bool) -> IntTensor<B> {
        // Default implementation. Backends can sort on the int values since qparams remain the same.
        let tensor_f = Self::dequantize(tensor);
        B::float_argsort(tensor_f, dim, descending)
    }
}
