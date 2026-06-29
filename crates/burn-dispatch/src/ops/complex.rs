use alloc::vec::Vec;
use burn_backend::{
    BoolDType, ComplexDType, ExecutionError, Scalar, Shape, Slice, TensorData, TensorMetadata,
    ops::ComplexTensorOps,
    tensor::{BoolTensor, ComplexTensor, FloatTensor, IntTensor},
};

//used in complex_to_device
#[allow(unused_imports)]
use burn_backend::ComplexTensorBackend;

use crate::{Dispatch, DispatchDevice};

impl ComplexTensorOps<Self> for Dispatch {
    fn complex_device(tensor: &burn_backend::ComplexTensor<Self>) -> DispatchDevice {
        tensor.device()
    }

    fn complex_random(
        shape: Shape,
        distribution: burn_backend::Distribution,
        device: &DispatchDevice,
        dtype: ComplexDType,
    ) -> ComplexTensor<Self> {
        creation_op!(Complex, device, |device| B::complex_random(
            shape,
            distribution,
            device,
            dtype
        ))
    }

    async fn complex_into_data(tensor: ComplexTensor<Self>) -> Result<TensorData, ExecutionError> {
        unary_complex!(tensor, complex, |tensor| B::complex_into_data(tensor).await)
    }

    fn complex_to_device(
        tensor: ComplexTensor<Self>,
        device: &DispatchDevice,
    ) -> ComplexTensor<Self> {
        to_device!(
            Complex,
            complex,
            tensor,
            device,
            complex_to_device,
            |inner, device| {
                let data = burn_backend::read_sync(B1::complex_into_data(inner))
                    .expect("Should read data");
                B2::complex_from_interleaved_data(data, device)
            }
        )
    }

    fn complex_add(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_add(lhs, rhs) => Complex)
    }

    fn complex_add_scalar(lhs: ComplexTensor<Self>, rhs: Scalar) -> ComplexTensor<Self> {
        unary_complex!(lhs, complex, |lhs| B::complex_add_scalar(lhs, rhs) => Complex)
    }

    fn complex_sub(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_sub(lhs, rhs) => Complex)
    }

    fn complex_sub_scalar(lhs: ComplexTensor<Self>, rhs: Scalar) -> ComplexTensor<Self> {
        unary_complex!(lhs, complex, |lhs| B::complex_sub_scalar(lhs, rhs) => Complex)
    }

    fn complex_mul(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_mul(lhs, rhs) => Complex)
    }

    fn complex_mul_scalar(lhs: ComplexTensor<Self>, rhs: Scalar) -> ComplexTensor<Self> {
        unary_complex!(lhs, complex, |lhs| B::complex_mul_scalar(lhs, rhs) => Complex)
    }

    fn complex_div(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_div(lhs, rhs) => Complex)
    }

    fn complex_div_scalar(lhs: ComplexTensor<Self>, rhs: Scalar) -> ComplexTensor<Self> {
        unary_complex!(lhs, complex, |lhs| B::complex_div_scalar(lhs, rhs) => Complex)
    }

    fn complex_remainder(
        lhs: ComplexTensor<Self>,
        rhs: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_remainder(lhs, rhs) => Complex)
    }

    fn complex_remainder_scalar(lhs: ComplexTensor<Self>, rhs: Scalar) -> ComplexTensor<Self> {
        unary_complex!(lhs, complex, |lhs| B::complex_remainder_scalar(lhs, rhs) => Complex)
    }

    fn complex_matmul(lhs: ComplexTensor<Self>, rhs: ComplexTensor<Self>) -> ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_matmul(lhs, rhs) => Complex)
    }

    fn complex_swap_dims(
        tensor: ComplexTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_swap_dims(tensor, dim1, dim2) => Complex)
    }

    fn complex_permute(tensor: ComplexTensor<Self>, axes: &[usize]) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_permute(tensor, axes) => Complex)
    }

    fn complex_flip(tensor: ComplexTensor<Self>, axes: &[usize]) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_flip(tensor, axes) => Complex)
    }

    fn complex_reshape(tensor: ComplexTensor<Self>, shape: Shape) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_reshape(tensor, shape) => Complex)
    }

    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<Self>,
        indices: IntTensor<Self>,
    ) -> ComplexTensor<Self> {
        binary_complex!((tensor, complex), (indices, int), |tensor, indices| B::complex_gather(dim, tensor, indices) => Complex)
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<Self>,
        indices: IntTensor<Self>,
        value: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        multi_op!(
            inputs[(tensor, complex), (indices, int), (value, complex)], => Complex,
            B::complex_scatter_add(dim, tensor, indices, value)
        )
    }

    fn complex_scatter_nd(
        data: ComplexTensor<Self>,
        indices: IntTensor<Self>,
        values: ComplexTensor<Self>,
        reduction: burn_backend::tensor::IndexingUpdateOp,
    ) -> ComplexTensor<Self> {
        multi_op!(
            inputs[(data, complex), (indices, int), (values, complex)], => Complex,
            B::complex_scatter_nd(data, indices, values, reduction)
        )
    }

    fn complex_gather_nd(
        data: ComplexTensor<Self>,
        indices: IntTensor<Self>,
    ) -> ComplexTensor<Self> {
        binary_complex!((data, complex), (indices, int), |data, indices| B::complex_gather_nd(data, indices) => Complex)
    }

    fn complex_select(
        tensor: ComplexTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> ComplexTensor<Self> {
        binary_complex!((tensor, complex), (indices, int), |tensor, indices| B::complex_select(tensor, dim, indices) => Complex)
    }

    fn complex_select_add(
        tensor: ComplexTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        multi_op!(
            inputs[(tensor, complex), (indices, int), (value, complex)], => Complex,
            B::complex_select_add(tensor, dim, indices, value)
        )
    }

    fn complex_slice(tensor: ComplexTensor<Self>, slices: &[Slice]) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_slice(tensor, slices) => Complex)
    }

    fn complex_slice_assign(
        tensor: ComplexTensor<Self>,
        slices: &[Slice],
        value: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        binary_complex!((tensor, complex), (value, complex), |tensor, value| B::complex_slice_assign(tensor, slices, value) => Complex)
    }

    fn complex_mask_where(
        tensor: ComplexTensor<Self>,
        mask: BoolTensor<Self>,
        value: ComplexTensor<Self>,
    ) -> ComplexTensor<Self> {
        multi_op!(
            inputs[(tensor, complex), (mask, bool), (value, complex)], => Complex,
            B::complex_mask_where(tensor, mask, value)
        )
    }

    fn complex_mask_fill(
        tensor: ComplexTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> ComplexTensor<Self> {
        binary_complex!((tensor, complex), (mask, bool), |tensor, mask| B::complex_mask_fill(tensor, mask, value) => Complex)
    }

    fn complex_equal(
        lhs: ComplexTensor<Self>,
        rhs: ComplexTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_equal(lhs, rhs, out_dtype) => Bool)
    }

    fn complex_equal_elem(
        lhs: ComplexTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unary_complex!(lhs, complex, |lhs| B::complex_equal_elem(lhs, rhs, out_dtype) => Bool)
    }

    fn complex_sum(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_sum(tensor) => Complex)
    }

    fn complex_sum_dim(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_sum_dim(tensor, dim) => Complex)
    }

    fn complex_mean_dim(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_mean_dim(tensor, dim) => Complex)
    }

    fn complex_cumsum(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_cumsum(tensor, dim) => Complex)
    }

    fn complex_cumprod(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_cumprod(tensor, dim) => Complex)
    }

    fn complex_exp(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_exp(tensor) => Complex)
    }

    fn complex_log(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_log(tensor) => Complex)
    }

    fn complex_powf(lhs: ComplexTensor<Self>, rhs: FloatTensor<Self>) -> ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, float), |lhs, rhs| B::complex_powf(lhs, rhs) => Complex)
    }

    fn complex_sqrt(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_sqrt(tensor) => Complex)
    }

    fn complex_cos(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_cos(tensor) => Complex)
    }

    fn complex_sin(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_sin(tensor) => Complex)
    }

    fn complex_tan(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_tan(tensor) => Complex)
    }

    fn complex_cosh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_cosh(tensor) => Complex)
    }

    fn complex_cast(tensor: ComplexTensor<Self>, dtype: ComplexDType) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_cast(tensor, dtype) => Complex)
    }

    fn complex_sinh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_sinh(tensor) => Complex)
    }

    fn complex_tanh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_tanh(tensor) => Complex)
    }

    fn complex_acos(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_acos(tensor) => Complex)
    }

    fn complex_acosh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_acosh(tensor) => Complex)
    }

    fn complex_asin(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_asin(tensor) => Complex)
    }

    fn complex_asinh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_asinh(tensor) => Complex)
    }

    fn complex_atan(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_atan(tensor) => Complex)
    }

    fn complex_atanh(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_atanh(tensor) => Complex)
    }

    fn complex_expand(tensor: ComplexTensor<Self>, shape: Shape) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_expand(tensor, shape) => Complex)
    }

    fn complex_atan2(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_atan2(lhs, rhs) => Complex)
    }

    fn complex_unfold(
        tensor: ComplexTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| {
            B::complex_unfold(tensor, dim, size, step)
        } => Complex)
    }

    // Default implementation
    fn complex_zeros(
        shape: Shape,
        device: &DispatchDevice,
        dtype: ComplexDType,
    ) -> ComplexTensor<Self> {
        complex_creation_op!(Complex, device, |device| B::complex_zeros(
            shape, device, dtype
        ))
    }

    fn complex_ones(
        shape: Shape,
        device: &DispatchDevice,
        dtype: ComplexDType,
    ) -> ComplexTensor<Self> {
        complex_creation_op!(Complex, device, |device| B::complex_ones(
            shape, device, dtype
        ))
    }

    fn complex_full(
        shape: Shape,
        fill_value: Scalar,
        device: &DispatchDevice,
        dtype: ComplexDType,
    ) -> ComplexTensor<Self> {
        complex_creation_op!(Complex, device, |device| B::complex_full(
            shape, fill_value, device, dtype
        ))
    }

    fn complex_repeat_dim(
        tensor: ComplexTensor<Self>,
        dim: usize,
        times: usize,
    ) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_repeat_dim(tensor, dim, times) => Complex)
    }

    fn complex_neg(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_neg(tensor) => Complex)
    }

    fn complex_transpose(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_transpose(tensor) => Complex)
    }

    fn complex_not_equal(
        lhs: ComplexTensor<Self>,
        rhs: ComplexTensor<Self>,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_not_equal(lhs, rhs, out_dtype) => Bool)
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<Self>,
        rhs: Scalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unary_complex!(lhs, complex, |lhs| B::complex_not_equal_elem(lhs, rhs, out_dtype) => Bool)
    }

    fn complex_prod(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_prod(tensor) => Complex)
    }

    fn complex_prod_dim(tensor: ComplexTensor<Self>, dim: usize) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_prod_dim(tensor, dim) => Complex)
    }

    fn complex_mean(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_mean(tensor) => Complex)
    }

    fn complex_powi(lhs: ComplexTensor<Self>, rhs: IntTensor<Self>) -> ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, int), |lhs, rhs| B::complex_powi(lhs, rhs) => Complex)
    }

    fn complex_powf_scalar(tensor: ComplexTensor<Self>, value: Scalar) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_powf_scalar(tensor, value) => Complex)
    }

    fn complex_cat(tensors: Vec<ComplexTensor<Self>>, dim: usize) -> ComplexTensor<Self> {
        vec_op!(tensors, complex, |tensors| B::complex_cat(tensors, dim) => Complex)
    }

    fn complex_any(tensor: ComplexTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_any(tensor, out_dtype) => Bool)
    }

    fn complex_any_dim(
        tensor: ComplexTensor<Self>,
        dim: usize,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_any_dim(tensor, dim, out_dtype) => Bool)
    }

    fn complex_all(tensor: ComplexTensor<Self>, out_dtype: BoolDType) -> BoolTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_all(tensor, out_dtype) => Bool)
    }

    fn complex_all_dim(
        tensor: ComplexTensor<Self>,
        dim: usize,
        out_dtype: BoolDType,
    ) -> BoolTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_all_dim(tensor, dim, out_dtype) => Bool)
    }

    fn complex_sign(tensor: ComplexTensor<Self>) -> ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_sign(tensor) => Complex)
    }

    async fn complex_into_interleaved_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<TensorData, ExecutionError> {
        unary_complex!(tensor, complex, |tensor| B::complex_into_interleaved_data(
            tensor
        )
        .await)
    }

    async fn complex_into_split_data(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> Result<(TensorData, TensorData), ExecutionError> {
        unary_complex!(tensor, complex, |tensor| B::complex_into_split_data(tensor)
            .await)
    }

    // fn to_complex(
    //     tensor: burn_backend::tensor::FloatTensor<Self>,
    // ) -> burn_backend::ComplexTensor<Self> {
    //     unary_complex!(tensor, complex, |tensor| B::to_complex(tensor))
    // }

    fn complex_squared_norm(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_squared_norm(tensor) => Float)
    }

    fn complex_conj(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_conj(tensor) => Complex)
    }

    fn complex_real(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_real(tensor) => Float)
    }

    fn complex_imag(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_imag(tensor) => Float)
    }

    fn complex_abs(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_abs(tensor) => Float)
    }

    fn complex_arg(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::tensor::FloatTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_arg(tensor) => Float)
    }

    fn complex_from_parts(
        real: TensorData,
        imag: TensorData,
        device: &DispatchDevice,
    ) -> burn_backend::ComplexTensor<Self> {
        complex_creation_op!(Complex, device, |device| B::complex_from_parts(
            real, imag, device
        ))
    }

    fn complex_from_polar(
        magnitude: burn_backend::tensor::FloatTensor<Self>,
        phase: burn_backend::tensor::FloatTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        binary_float!((magnitude, float), (phase, float), |magnitude, phase| B::complex_from_polar(magnitude, phase) => Complex)
    }

    fn complex_powc(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        binary_complex!((lhs, complex), (rhs, complex), |lhs, rhs| B::complex_powc(lhs, rhs) => Complex)
    }

    fn complex_powc_scalar(
        lhs: burn_backend::ComplexTensor<Self>,
        rhs: Scalar,
    ) -> burn_backend::ComplexTensor<Self> {
        unary_complex!(lhs, complex, |lhs| B::complex_powc_scalar(lhs, rhs) => Complex)
    }

    fn complex_recip(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_recip(tensor) => Complex)
    }

    fn complex_finv(
        tensor: burn_backend::ComplexTensor<Self>,
    ) -> burn_backend::ComplexTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_finv(tensor) => Complex)
    }

    fn complex_into_float(
        tensor: burn_backend::ComplexTensor<Self>,
        dtype: burn_backend::FloatDType,
    ) -> FloatTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_into_float(tensor, dtype) => Float)
    }

    fn complex_into_int(
        tensor: burn_backend::ComplexTensor<Self>,
        dtype: burn_backend::IntDType,
    ) -> IntTensor<Self> {
        unary_complex!(tensor, complex, |tensor| B::complex_into_int(tensor, dtype) => Int)
    }
}
