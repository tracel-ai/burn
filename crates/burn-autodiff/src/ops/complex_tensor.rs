use burn_backend::{
    Backend, ComplexTensorBackend,
    ops::ComplexTensorOps,
    tensor::{Device, FloatTensor, IntTensor},
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor};

impl<B: ComplexTensorBackend + Backend, C: CheckpointStrategy> ComplexTensorBackend
    for Autodiff<B, C>
{
}
impl<B: ComplexTensorBackend + Backend, C: CheckpointStrategy> ComplexTensorOps<Self>
    for Autodiff<B, C>
{
    fn complex_device(tensor: &burn_backend::ComplexTensor<Self>) -> B::Device {
        B::complex_device(tensor)
    }

    fn complex_from_data(
        data: burn_std::TensorData,
        device: &Device<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_from_data(data, device)
    }

    fn complex_from_parts_data(
        real_data: burn_std::TensorData,
        imag_data: burn_std::TensorData,
        device: &Device<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_from_parts_data(real_data, imag_data, device)
    }

    async fn complex_into_interleaved_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<burn_std::TensorData, burn_std::ExecutionError> {
        B::complex_into_interleaved_data(tensor).await
    }

    async fn complex_into_split_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<(burn_std::TensorData, burn_std::TensorData), burn_std::ExecutionError> {
        B::complex_into_split_data(tensor).await
    }

    fn complex_squared_norm(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        AutodiffTensor::new(B::complex_squared_norm(tensor))
    }

    fn complex_random(
        shape: burn_std::Shape,
        distribution: burn_std::Distribution,
        device: &Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_random(shape, distribution, device, dtype)
    }

    fn complex_zeros(
        shape: burn_std::Shape,
        device: &Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_zeros(shape, device, dtype)
    }

    fn complex_empty(
        shape: burn_std::Shape,
        device: &Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_empty(shape, device, dtype)
    }

    fn complex_ones(
        shape: burn_std::Shape,
        device: &Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_ones(shape, device, dtype)
    }

    fn complex_full(
        shape: burn_std::Shape,
        fill_value: burn_std::Scalar,
        device: &Device<Self>,
        dtype: burn_std::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_full(shape, fill_value, device, dtype)
    }

    fn complex_to_device(
        tensor: burn_backend::ComplexTensor<Self>,
        device: &B::Device,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_to_device(tensor, device)
    }

    async fn complex_into_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<burn_std::TensorData, burn_std::ExecutionError> {
        B::complex_into_data(tensor).await
    }

    fn complex_reshape(
        tensor: burn_backend::ComplexTensor<Self>,
        shape: burn_std::Shape,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_reshape(tensor, shape)
    }

    fn complex_add(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_add(lhs, rhs)
    }

    fn complex_sub(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_sub(lhs, rhs)
    }

    fn complex_mul(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_mul(lhs, rhs)
    }

    fn complex_div(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_div(lhs, rhs)
    }

    fn complex_neg(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        B::complex_neg(tensor)
    }

    fn complex_conj(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_conj(tensor)
    }

    fn complex_recip(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_recip(tensor)
    }

    fn complex_finv(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_finv(tensor)
    }

    fn complex_real(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        AutodiffTensor::new(B::complex_real(tensor))
    }

    fn complex_imag(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        AutodiffTensor::new(B::complex_imag(tensor))
    }

    fn complex_into_float(
        tensor: burn_backend::ComplexTensor<Self>,
        dtype: burn_std::FloatDType,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        AutodiffTensor::new(B::complex_into_float(tensor, dtype))
    }

    fn complex_into_int(
        tensor: burn_backend::ComplexTensor<Self>,
        dtype: burn_std::IntDType,
    ) -> IntTensor<Self> {
        B::complex_into_int(tensor, dtype)
    }

    fn complex_abs(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        AutodiffTensor::new(B::complex_abs(tensor))
    }

    fn complex_arg(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        AutodiffTensor::new(B::complex_arg(tensor))
    }

    fn complex_from_parts(
        real: FloatTensor<Self>,
        imag: FloatTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_from_parts(real.primitive, imag.primitive)
    }

    fn complex_from_polar(
        magnitude: burn_backend::tensor::FloatTensor<Self>,
        phase: burn_backend::tensor::FloatTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_from_polar(magnitude.primitive, phase.primitive)
    }

    fn complex_exp(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        B::complex_exp(tensor)
    }

    fn complex_log(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        B::complex_log(tensor)
    }

    fn complex_powc(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_powc(lhs, rhs)
    }

    fn complex_sqrt(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_sqrt(tensor)
    }

    fn complex_sin(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        B::complex_sin(tensor)
    }

    fn complex_cos(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        B::complex_cos(tensor)
    }

    fn complex_tan(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        B::complex_tan(tensor)
    }

    fn complex_acos(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_acos(tensor)
    }

    fn complex_acosh(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_acosh(tensor)
    }

    fn complex_cast(
        tensor: burn_backend::ComplexTensor<Self>,
        dtype: burn_std::ComplexDType,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_cast(tensor, dtype)
    }

    fn complex_asin(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_asin(tensor)
    }

    fn complex_asinh(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_asinh(tensor)
    }

    fn complex_atan(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_atan(tensor)
    }

    fn complex_atanh(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_atanh(tensor)
    }

    fn complex_atan2(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_atan2(lhs, rhs)
    }

    fn complex_select(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_select(tensor, dim, indices)
    }

    fn complex_slice(
        tensor: burn_backend::ComplexTensor<Self>,
        slices: &[burn_std::Slice],
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_slice(tensor, slices)
    }

    fn complex_slice_assign(
        tensor: burn_backend::ComplexTensor<Self>,
        ranges: &[burn_std::Slice],
        value: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_slice_assign(tensor, ranges, value)
    }

    fn complex_scatter_nd(
        tensor: burn_backend::ComplexTensor<Self>,
        indices: B::IntTensorPrimitive,
        value: burn_backend::ComplexTensor<Self>,
        reduction: burn_std::IndexingUpdateOp,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_scatter_nd(tensor, indices, value, reduction)
    }

    fn complex_swap_dims(
        tensor: burn_backend::ComplexTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_swap_dims(tensor, dim1, dim2)
    }

    fn complex_repeat_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        times: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_repeat_dim(tensor, dim, times)
    }

    fn complex_equal(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        B::complex_equal(lhs, rhs, out_dtype)
    }

    fn complex_not_equal(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        B::complex_not_equal(lhs, rhs, out_dtype)
    }

    fn complex_cat(
        tensors: alloc::vec::Vec<burn_backend::ComplexTensor<Self>>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_cat(tensors, dim)
    }

    fn complex_any(
        tensor: burn_backend::ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        B::complex_any(tensor, out_dtype)
    }

    fn complex_any_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        B::complex_any_dim(tensor, dim, out_dtype)
    }

    fn complex_all(
        tensor: burn_backend::ComplexTensor<Self>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        B::complex_all(tensor, out_dtype)
    }

    fn complex_all_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        B::complex_all_dim(tensor, dim, out_dtype)
    }

    fn complex_permute(
        tensor: burn_backend::ComplexTensor<Self>,
        axes: &[usize],
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_permute(tensor, axes)
    }

    fn complex_expand(
        tensor: burn_backend::ComplexTensor<Self>,
        shape: burn_std::Shape,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_expand(tensor, shape)
    }

    fn complex_flip(
        tensor: burn_backend::ComplexTensor<Self>,
        axes: &[usize],
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_flip(tensor, axes)
    }

    fn complex_unfold(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_unfold(tensor, dim, size, step)
    }

    fn complex_select_add(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_select_add(tensor, dim, indices, values)
    }

    fn complex_sum(tensor: burn_backend::ComplexTensor<Self>) -> burn_backend::ComplexTensor<Self> {
        B::complex_sum(tensor)
    }

    fn complex_sum_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_sum_dim(tensor, dim)
    }

    fn complex_prod(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_prod(tensor)
    }

    fn complex_prod_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_prod_dim(tensor, dim)
    }

    fn complex_mean(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_mean(tensor)
    }

    fn complex_mean_dim(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_mean_dim(tensor, dim)
    }

    fn complex_remainder(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_remainder(lhs, rhs)
    }

    fn complex_remainder_scalar(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_std::Scalar,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_remainder_scalar(lhs, rhs)
    }

    fn complex_equal_elem(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        B::complex_equal_elem(lhs, rhs, out_dtype)
    }

    fn complex_not_equal_elem(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_std::Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        B::complex_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn complex_mask_where(
        tensor: burn_backend::ComplexTensor<Self>,
        mask: B::BoolTensorPrimitive,
        source: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_mask_where(tensor, mask, source)
    }

    fn complex_mask_fill(
        tensor: burn_backend::ComplexTensor<Self>,
        mask: B::BoolTensorPrimitive,
        value: burn_std::Scalar,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_mask_fill(tensor, mask, value)
    }

    fn complex_gather(
        dim: usize,
        tensor: burn_backend::ComplexTensor<Self>,
        indices: B::IntTensorPrimitive,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_gather(dim, tensor, indices)
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: burn_backend::ComplexTensor<Self>,
        indices: B::IntTensorPrimitive,
        values: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_scatter_add(dim, tensor, indices, values)
    }

    fn complex_sign(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_sign(tensor)
    }

    fn complex_powc_scalar(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_std::Scalar,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_powc_scalar(lhs, rhs)
    }

    fn complex_powf(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::tensor::FloatTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_powf(lhs, rhs.primitive)
    }

    fn complex_powf_scalar(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_std::Scalar,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_powf_scalar(lhs, rhs)
    }

    fn complex_matmul(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_matmul(lhs, rhs)
    }

    fn complex_cumsum(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_cumsum(tensor, dim)
    }

    fn complex_cumprod(
        tensor: burn_backend::ComplexTensor<Self>,
        dim: usize,
    ) -> burn_backend::ComplexTensor<Self> {
        B::complex_cumprod(tensor, dim)
    }
}
