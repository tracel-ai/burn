use alloc::vec::Vec;
use burn_backend::{
    BoolDType, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice, TensorData,
    ops::{FloatTensorOps, PadMode},
    tensor::{BoolTensor, FloatTensor, IndexingUpdateOp, IntTensor},
};
use burn_backend_extension::backend_dispatch;

use crate::{Dispatch, DispatchDevice};

#[backend_dispatch]
impl FloatTensorOps<Self> for Dispatch {
    fn float_pad(
        tensor: FloatTensor<Self>,
        padding: &[(usize, usize)],
        mode: PadMode,
    ) -> FloatTensor<Self> {
        B::float_pad(tensor, padding, mode)
    }

    fn float_from_data(
        data: burn_backend::TensorData,
        device: &DispatchDevice,
    ) -> FloatTensor<Self> {
        B::float_from_data(data, device)
    }

    fn float_random(
        shape: Shape,
        distribution: burn_backend::Distribution,
        device: &DispatchDevice,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        B::float_random(shape, distribution, device, dtype)
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        B::float_into_data(tensor).await
    }

    fn float_svd(
        tensor: FloatTensor<Self>,
        sweeps: usize,
        swap: bool,
    ) -> (FloatTensor<Self>, FloatTensor<Self>, FloatTensor<Self>) {
        B::float_svd(tensor, sweeps, swap)
    }

    #[backend_dispatch(skip)]
    fn float_to_device(tensor: FloatTensor<Self>, device: &DispatchDevice) -> FloatTensor<Self> {
        // Relocating a non-tracked float tensor onto an autodiff device is a plain data move:
        // place it on the underlying hardware device and leave the tensor non-tracked. The
        // int/bool `to_device` paths already handle this case; only the float path used to
        // panic. This is what lets gradient tensors — which are never autodiff-tracked — be
        // moved onto the autodiff `device_main` during multi-device training.
        #[cfg(feature = "autodiff")]
        if let DispatchDevice::Autodiff(device_ad) = device
            && !matches!(&tensor.kind, crate::DispatchTensorKind::Autodiff(_))
        {
            return Self::float_to_device(tensor, &device_ad.inner);
        }

        float_to_device!(
            Float,
            float,
            tensor,
            device,
            float_to_device,
            |inner, device| {
                let data =
                    burn_backend::read_sync(B1::float_into_data(inner)).expect("Should read data");
                B2::float_from_data(data, device)
            }
        )
    }

    fn float_into_int(tensor: FloatTensor<Self>, dtype: burn_backend::IntDType) -> IntTensor<Self> {
        B::float_into_int(tensor, dtype)
    }

    fn float_empty(shape: Shape, device: &DispatchDevice, dtype: FloatDType) -> FloatTensor<Self> {
        B::float_empty(shape, device, dtype)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_add(lhs, rhs)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        B::float_add_scalar(lhs, rhs)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_sub(lhs, rhs)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        B::float_sub_scalar(lhs, rhs)
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_mul(lhs, rhs)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        B::float_mul_scalar(lhs, rhs)
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_div(lhs, rhs)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        B::float_div_scalar(lhs, rhs)
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_remainder(lhs, rhs)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        B::float_remainder_scalar(lhs, rhs)
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_matmul(lhs, rhs)
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        B::float_cross(lhs, rhs, dim)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_recip(tensor)
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        B::float_swap_dims(tensor, dim1, dim2)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        B::float_permute(tensor, axes)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        B::float_flip(tensor, axes)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        B::float_reshape(tensor, shape)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        B::float_gather(dim, tensor, indices)
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
        update: IndexingUpdateOp,
    ) -> FloatTensor<Self> {
        B::float_scatter(dim, tensor, indices, value, update)
    }

    fn float_scatter_nd(
        data: FloatTensor<Self>,
        indices: IntTensor<Self>,
        values: FloatTensor<Self>,
        reduction: burn_backend::tensor::IndexingUpdateOp,
    ) -> FloatTensor<Self> {
        B::float_scatter_nd(data, indices, values, reduction)
    }

    fn float_gather_nd(data: FloatTensor<Self>, indices: IntTensor<Self>) -> FloatTensor<Self> {
        B::float_gather_nd(data, indices)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        B::float_select(tensor, dim, indices)
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
        update: IndexingUpdateOp,
    ) -> FloatTensor<Self> {
        B::float_select_assign(tensor, dim, indices, value, update)
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        B::float_slice(tensor, slices)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::float_slice_assign(tensor, slices, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        B::float_mask_where(tensor, mask, value)
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> FloatTensor<Self> {
        B::float_mask_fill(tensor, mask, value)
    }

    async fn float_mask_select(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
    ) -> FloatTensor<Self> {
        B::float_mask_select(tensor, mask).await
    }

    fn float_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_equal(lhs, rhs, out_dtype)
    }

    fn float_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_greater(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_greater(lhs, rhs, out_dtype)
    }

    fn float_greater_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_greater_elem(lhs, rhs, out_dtype)
    }

    fn float_greater_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_greater_equal(lhs, rhs, out_dtype)
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_lower(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_lower(lhs, rhs, out_dtype)
    }

    fn float_lower_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_lower_elem(lhs, rhs, out_dtype)
    }

    fn float_lower_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_lower_equal(lhs, rhs, out_dtype)
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_sum(tensor)
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_sum_dim(tensor, dim)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_mean_dim(tensor, dim)
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_cumsum(tensor, dim)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_cumprod(tensor, dim)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_cummin(tensor, dim)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_cummax(tensor, dim)
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        B::float_cast(tensor, dtype)
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_exp(tensor)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_log(tensor)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_log1p(tensor)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_powf(lhs, rhs)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        B::float_powf_scalar_impl(tensor, value)
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_sqrt(tensor)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_abs(tensor)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_cos(tensor)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_sin(tensor)
    }

    fn float_tan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_tan(tensor)
    }

    fn float_cosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_cosh(tensor)
    }

    fn float_sinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_sinh(tensor)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_tanh(tensor)
    }

    fn float_acos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_acos(tensor)
    }

    fn float_acosh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_acosh(tensor)
    }

    fn float_asin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_asin(tensor)
    }

    fn float_asinh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_asinh(tensor)
    }

    fn float_atan(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_atan(tensor)
    }

    fn float_atanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_atanh(tensor)
    }

    fn float_atan2(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_atan2(lhs, rhs)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_round(tensor)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_floor(tensor)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_ceil(tensor)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_trunc(tensor)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_erf(tensor)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::float_argmax(tensor, dim, out_dtype)
    }

    fn float_argtopk(
        tensor: FloatTensor<Self>,
        dim: usize,
        k: usize,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        B::float_argtopk(tensor, dim, k, out_dtype)
    }

    fn float_topk(tensor: FloatTensor<Self>, dim: usize, k: usize) -> FloatTensor<Self> {
        B::float_topk(tensor, dim, k)
    }

    fn float_topk_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
        k: usize,
        out_dtype: IntDType,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        B::float_topk_with_indices(tensor, dim, k, out_dtype)
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::float_argmin(tensor, dim, out_dtype)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        B::float_expand(tensor, shape)
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        B::float_unfold(tensor, dim, size, step)
    }

    fn float_detach(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_detach(tensor)
    }

    fn float_set_require_grad(tensor: FloatTensor<Self>, require_grad: bool) -> FloatTensor<Self> {
        B::float_set_require_grad(tensor, require_grad)
    }

    fn float_is_require_grad(tensor: &FloatTensor<Self>) -> bool {
        B::float_is_require_grad(tensor)
    }

    // Default implementation
    fn float_zeros(shape: Shape, device: &DispatchDevice, dtype: FloatDType) -> FloatTensor<Self> {
        B::float_zeros(shape, device, dtype)
    }

    fn float_ones(shape: Shape, device: &DispatchDevice, dtype: FloatDType) -> FloatTensor<Self> {
        B::float_ones(shape, device, dtype)
    }

    fn float_full(
        shape: Shape,
        fill_value: Scalar,
        device: &DispatchDevice,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        B::float_full(shape, fill_value, device, dtype)
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        B::float_repeat_dim(tensor, dim, times)
    }

    fn float_clamp_min(tensor: FloatTensor<Self>, min: Scalar) -> FloatTensor<Self> {
        B::float_clamp_min(tensor, min)
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: Scalar) -> FloatTensor<Self> {
        B::float_clamp_max(tensor, max)
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: Scalar, max: Scalar) -> FloatTensor<Self> {
        B::float_clamp(tensor, min, max)
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_neg(tensor)
    }

    fn float_transpose(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_transpose(tensor)
    }

    fn float_not_equal(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_not_equal(lhs, rhs, out_dtype)
    }

    fn float_not_equal_elem(
        lhs: FloatTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_prod(tensor)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_prod_dim(tensor, dim)
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_mean(tensor)
    }

    fn float_powi(lhs: FloatTensor<Self>, rhs: IntTensor<Self>) -> FloatTensor<Self> {
        B::float_powi(lhs, rhs)
    }

    fn float_powi_scalar_impl(lhs: FloatTensor<Self>, rhs: Scalar) -> FloatTensor<Self> {
        B::float_powi_scalar_impl(lhs, rhs)
    }

    fn float_powf_scalar(tensor: FloatTensor<Self>, value: Scalar) -> FloatTensor<Self> {
        B::float_powf_scalar(tensor, value)
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        B::float_cat(tensors, dim)
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_max(tensor)
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_max_dim(tensor, dim)
    }

    fn float_max_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices_dtype: IntDType,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        B::float_max_dim_with_indices(tensor, dim, indices_dtype)
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_min(tensor)
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_min_dim(tensor, dim)
    }

    fn float_min_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices_dtype: IntDType,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        B::float_min_dim_with_indices(tensor, dim, indices_dtype)
    }

    fn float_max_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_max_abs(tensor)
    }

    fn float_max_abs_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        B::float_max_abs_dim(tensor, dim)
    }

    fn float_any(tensor: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        B::float_any(tensor, out_dtype)
    }

    fn float_any_dim(
        tensor: FloatTensor<Self>,
        dim: usize,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_any_dim(tensor, dim, out_dtype)
    }

    fn float_all(tensor: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        B::float_all(tensor, out_dtype)
    }

    fn float_all_dim(
        tensor: FloatTensor<Self>,
        dim: usize,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        B::float_all_dim(tensor, dim, out_dtype)
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_sign(tensor)
    }

    fn float_sort(tensor: FloatTensor<Self>, dim: usize, descending: bool) -> FloatTensor<Self> {
        B::float_sort(tensor, dim, descending)
    }

    fn float_sort_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
        descending: bool,
        indices_dtype: IntDType,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        B::float_sort_with_indices(tensor, dim, descending, indices_dtype)
    }

    fn float_argsort(
        tensor: FloatTensor<Self>,
        dim: usize,
        descending: bool,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        B::float_argsort(tensor, dim, descending, out_dtype)
    }

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: burn_backend::ops::GridSampleOptions,
    ) -> FloatTensor<Self> {
        B::float_grid_sample_2d(tensor, grid, options)
    }

    fn float_is_nan(tensor: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        B::float_is_nan(tensor, out_dtype)
    }

    fn float_is_inf(tensor: FloatTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        B::float_is_inf(tensor, out_dtype)
    }

    fn float_hypot(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        B::float_hypot(lhs, rhs)
    }
}

#[cfg(all(test, feature = "capture", any(feature = "flex", default_backend)))]
mod tests {
    use super::*;
    use burn_backend::ops::{BoolTensorOps, IntTensorOps};
    use burn_capture::CaptureDevice;

    #[test]
    fn capture_tensor_movement_is_one_way() {
        let source_device = DispatchDevice::Flex(Default::default());
        let concrete_capture_device = CaptureDevice::default();
        let capture_device = DispatchDevice::Capture(concrete_capture_device);
        let float = Dispatch::float_from_data(TensorData::from([1.0f32, 2.0]), &source_device);
        let int = Dispatch::int_from_data(TensorData::from([1i64, 2]), &source_device);

        let graph = concrete_capture_device
            .capture_scope(|scope| {
                let captured_float = Dispatch::float_to_device(float, &capture_device);
                let captured_float = Dispatch::float_to_device(captured_float, &capture_device);
                let captured_int = Dispatch::int_to_device(int, &capture_device);
                let captured_int = Dispatch::int_to_device(captured_int, &capture_device);
                let float_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Dispatch::float_to_device(captured_float, &source_device)
                }));
                let int_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    Dispatch::int_to_device(captured_int, &source_device)
                }));

                assert!(float_result.is_err());
                assert!(int_result.is_err());
                scope.complete([], [])
            })
            .unwrap();
        assert_eq!(graph.values.len(), 2);
    }

    #[test]
    fn initialized_tensors_can_move_between_capture_devices() {
        let source_device = DispatchDevice::Flex(Default::default());
        let first = CaptureDevice::default();
        let second = CaptureDevice::default();
        let first_dispatch = DispatchDevice::Capture(first);
        let second_dispatch = DispatchDevice::Capture(second);
        let float = Dispatch::float_from_data(TensorData::from([1.0f32]), &source_device);
        let int = Dispatch::int_from_data(TensorData::from([1i64]), &source_device);
        let bool = Dispatch::bool_from_data(TensorData::from([true]), &source_device);

        let first_graph = first
            .capture_scope(|first_scope| {
                let float = Dispatch::float_to_device(float, &first_dispatch);
                let int = Dispatch::int_to_device(int, &first_dispatch);
                let bool = Dispatch::bool_to_device(bool, &first_dispatch);

                let second_graph = second
                    .capture_scope(|second_scope| {
                        Dispatch::float_to_device(float, &second_dispatch);
                        Dispatch::int_to_device(int, &second_dispatch);
                        Dispatch::bool_to_device(bool, &second_dispatch);
                        second_scope.complete([], [])
                    })
                    .unwrap();
                assert_eq!(second_graph.values.len(), 3);

                first_scope.complete([], [])
            })
            .unwrap();

        assert_eq!(first_graph.values.len(), 3);
    }

    #[test]
    fn computed_tensors_cannot_move_between_capture_devices() {
        let source_device = DispatchDevice::Flex(Default::default());
        let first = CaptureDevice::default();
        let second = CaptureDevice::default();
        let first_dispatch = DispatchDevice::Capture(first);
        let second_dispatch = DispatchDevice::Capture(second);
        let float = Dispatch::float_from_data(TensorData::from([1.0f32]), &source_device);
        let int = Dispatch::int_from_data(TensorData::from([1i64]), &source_device);
        let bool = Dispatch::bool_from_data(TensorData::from([true]), &source_device);

        let first_graph = first
            .capture_scope(|first_scope| {
                let float = Dispatch::float_neg(Dispatch::float_to_device(float, &first_dispatch));
                let int = Dispatch::int_neg(Dispatch::int_to_device(int, &first_dispatch));
                let bool = Dispatch::bool_not(Dispatch::bool_to_device(bool, &first_dispatch));

                let second_graph = second
                    .capture_scope(|second_scope| {
                        let float_result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                Dispatch::float_to_device(float, &second_dispatch)
                            }));
                        let int_result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                Dispatch::int_to_device(int, &second_dispatch)
                            }));
                        let bool_result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                Dispatch::bool_to_device(bool, &second_dispatch)
                            }));

                        assert!(float_result.is_err());
                        assert!(int_result.is_err());
                        assert!(bool_result.is_err());
                        second_scope.complete([], [])
                    })
                    .unwrap();
                assert!(second_graph.values.is_empty());

                first_scope.complete([], [])
            })
            .unwrap();

        assert_eq!(first_graph.values.len(), 3);
    }
}
