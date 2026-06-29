use alloc::vec::Vec;
use burn_backend::{
    AutodiffBackend, Distribution, Scalar, TensorData, TensorMetadata,
    ops::{IntTensorOps, TransactionPrimitive},
};
use burn_dispatch::Dispatch;
use burn_std::{DType, ExecutionError, IndexingUpdateOp, Shape, Slice};

use crate::{
    Device, Int,
    bridge::{BasicAutodiffOps, BasicOps, Numeric, Ordered, TransactionOp},
    ops::BridgeTensor,
};

impl TransactionOp for Int {
    fn register_transaction(tr: &mut TransactionPrimitive<Dispatch>, tensor: BridgeTensor) {
        tr.register_int(tensor.into());
    }
}
impl BasicOps for Int {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_empty(
            shape,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_zeros(
            shape,
            device.as_dispatch(),
            dtype.into(),
        ))
    }
    fn ones(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_ones(
            shape,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_full(
            shape,
            fill_value,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn reshape(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_reshape(tensor.into(), shape))
    }

    fn transpose(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_transpose(tensor.into()))
    }

    fn swap_dims(tensor: BridgeTensor, dim1: usize, dim2: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_swap_dims(tensor.into(), dim1, dim2))
    }

    fn slice(tensor: BridgeTensor, slices: &[Slice]) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_slice(tensor.into(), slices))
    }

    fn slice_assign(tensor: BridgeTensor, slices: &[Slice], value: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_slice_assign(
            tensor.into(),
            slices,
            value.into(),
        ))
    }

    fn select(tensor: BridgeTensor, dim: usize, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_select(tensor.into(), dim, indices.into()))
    }

    fn select_assign(
        tensor: BridgeTensor,
        dim: usize,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        match update {
            IndexingUpdateOp::Add => BridgeTensor::int(Dispatch::int_select_add(
                tensor.into(),
                dim,
                indices.into(),
                values.into(),
            )),
            _ => unimplemented!(),
        }
    }

    fn mask_where(tensor: BridgeTensor, mask: BridgeTensor, source: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_mask_where(
            tensor.into(),
            mask.into(),
            source.into(),
        ))
    }

    fn mask_fill(tensor: BridgeTensor, mask: BridgeTensor, value: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_mask_fill(tensor.into(), mask.into(), value))
    }

    fn gather(dim: usize, tensor: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_gather(dim, tensor.into(), indices.into()))
    }

    fn scatter(
        dim: usize,
        tensor: BridgeTensor,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        match update {
            IndexingUpdateOp::Add => BridgeTensor::int(Dispatch::int_scatter_add(
                dim,
                tensor.into(),
                indices.into(),
                values.into(),
            )),
            _ => unimplemented!(),
        }
    }

    fn scatter_nd(
        data: BridgeTensor,
        indices: BridgeTensor,
        values: BridgeTensor,
        reduction: IndexingUpdateOp,
    ) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_scatter_nd(
            data.into(),
            indices.into(),
            values.into(),
            reduction,
        ))
    }

    fn gather_nd(data: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_gather_nd(data.into(), indices.into()))
    }

    fn device(tensor: &BridgeTensor) -> Device {
        Device::new(tensor.as_dispatch().device())
    }

    fn to_device(tensor: BridgeTensor, device: &Device) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_to_device(tensor.into(), device.as_dispatch()))
    }

    async fn into_data_async(tensor: BridgeTensor) -> Result<TensorData, ExecutionError> {
        Dispatch::int_into_data(tensor.into()).await
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_from_data(
            data.convert_dtype(dtype),
            device.as_dispatch(),
        ))
    }

    fn repeat_dim(tensor: BridgeTensor, dim: usize, times: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_repeat_dim(tensor.into(), dim, times))
    }

    fn equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::int(Dispatch::int_equal(lhs.into(), rhs.into(), bool_dtype))
    }

    fn not_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::int(Dispatch::int_not_equal(lhs.into(), rhs.into(), bool_dtype))
    }

    fn equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::int(Dispatch::int_equal_elem(lhs.into(), rhs, bool_dtype))
    }

    fn not_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::int(Dispatch::int_not_equal_elem(lhs.into(), rhs, bool_dtype))
    }

    fn cat(vectors: Vec<BridgeTensor>, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_cat(
            BridgeTensor::into_dispatch_vec(vectors),
            dim,
        ))
    }

    fn any(tensor: BridgeTensor) -> BridgeTensor {
        let settings = tensor.device_settings();
        BridgeTensor::int(Dispatch::int_any(tensor.into(), settings.bool_dtype))
    }

    fn any_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let bool_dtype = tensor.device_settings().bool_dtype;
        BridgeTensor::int(Dispatch::int_any_dim(tensor.into(), dim, bool_dtype))
    }

    fn all(tensor: BridgeTensor) -> BridgeTensor {
        let bool_dtype = tensor.device_settings().bool_dtype;
        BridgeTensor::int(Dispatch::int_all(tensor.into(), bool_dtype))
    }

    fn all_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let bool_dtype = tensor.device_settings().bool_dtype;
        BridgeTensor::int(Dispatch::int_all_dim(tensor.into(), dim, bool_dtype))
    }

    fn permute(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_permute(tensor.into(), axes))
    }

    fn expand(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_expand(tensor.into(), shape))
    }

    fn flip(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_flip(tensor.into(), axes))
    }

    fn unfold(tensor: BridgeTensor, dim: usize, size: usize, step: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_unfold(tensor.into(), dim, size, step))
    }
}

impl Numeric for Int {
    fn add(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_add(lhs.into(), rhs.into()))
    }
    fn add_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_add_scalar(lhs.into(), rhs))
    }
    fn sub(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_sub(lhs.into(), rhs.into()))
    }
    fn sub_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_sub_scalar(lhs.into(), rhs))
    }
    fn div(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_div(lhs.into(), rhs.into()))
    }
    fn div_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_div_scalar(lhs.into(), rhs))
    }
    fn remainder(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_remainder(lhs.into(), rhs.into()))
    }
    fn remainder_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_remainder_scalar(lhs.into(), rhs))
    }
    fn mul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_mul(lhs.into(), rhs.into()))
    }
    fn mul_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_mul_scalar(lhs.into(), rhs))
    }
    fn neg(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_neg(tensor.into()))
    }

    fn sum(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_sum(tensor.into()))
    }

    fn sum_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_sum_dim(tensor.into(), dim))
    }

    fn prod(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_prod(tensor.into()))
    }

    fn prod_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_prod_dim(tensor.into(), dim))
    }

    fn mean(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_mean(tensor.into()))
    }
    fn mean_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_mean_dim(tensor.into(), dim))
    }
    fn cumsum(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_cumsum(tensor.into(), dim))
    }
    fn cumprod(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_cumprod(tensor.into(), dim))
    }

    fn powi(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_powi(lhs.into(), rhs.into()))
    }

    fn powi_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_powi_scalar(lhs.into(), rhs))
    }

    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &Device,
        dtype: DType,
    ) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_random(
            shape,
            distribution,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn sign(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_sign(tensor.into()))
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors don't have a compatible shape.
    fn matmul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_matmul(lhs.into(), rhs.into()))
    }
}

impl Ordered for Int {
    fn abs(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_abs(tensor.into()))
    }
    fn sort(tensor: BridgeTensor, dim: usize, descending: bool) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_sort(tensor.into(), dim, descending))
    }

    fn sort_with_indices(
        tensor: BridgeTensor,
        dim: usize,
        descending: bool,
    ) -> (BridgeTensor, BridgeTensor) {
        let (values, indices) = Dispatch::int_sort_with_indices(tensor.into(), dim, descending);
        (BridgeTensor::int(values), BridgeTensor::int(indices))
    }

    fn argsort(tensor: BridgeTensor, dim: usize, descending: bool) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_argsort(tensor.into(), dim, descending))
    }

    fn cummin(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_cummin(tensor.into(), dim))
    }

    fn cummax(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_cummax(tensor.into(), dim))
    }

    fn greater(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::int(Dispatch::int_greater(lhs.into(), rhs.into(), bool_dtype))
    }

    fn greater_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::bool(Dispatch::int_greater_elem(lhs.into(), rhs, bool_dtype))
    }

    fn greater_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::bool(Dispatch::int_greater_equal(
            lhs.into(),
            rhs.into(),
            bool_dtype,
        ))
    }

    fn greater_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::bool(Dispatch::int_greater_equal_elem(
            lhs.into(),
            rhs,
            bool_dtype,
        ))
    }

    fn lower(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::bool(Dispatch::int_lower(lhs.into(), rhs.into(), bool_dtype))
    }

    fn lower_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::bool(Dispatch::int_lower_elem(lhs.into(), rhs, bool_dtype))
    }

    fn lower_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::bool(Dispatch::int_lower_equal(
            lhs.into(),
            rhs.into(),
            bool_dtype,
        ))
    }

    fn lower_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let bool_dtype = lhs.device_settings().bool_dtype;
        BridgeTensor::bool(Dispatch::int_lower_equal_elem(lhs.into(), rhs, bool_dtype))
    }

    fn argmax(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_argmax(tensor.into(), dim))
    }

    fn argtopk(tensor: BridgeTensor, dim: usize, k: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_argtopk(tensor.into(), dim, k))
    }

    fn topk(tensor: BridgeTensor, dim: usize, k: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_topk(tensor.into(), dim, k))
    }

    fn argmin(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_argmin(tensor.into(), dim))
    }

    fn max(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_max(tensor.into()))
    }

    fn max_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_max_dim(tensor.into(), dim))
    }

    fn max_dim_with_indices(tensor: BridgeTensor, dim: usize) -> (BridgeTensor, BridgeTensor) {
        let (values, indices) = Dispatch::int_max_dim_with_indices(tensor.into(), dim);
        (BridgeTensor::int(values), BridgeTensor::int(indices))
    }

    fn max_abs(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_max_abs(tensor.into()))
    }

    fn max_abs_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_max_abs_dim(tensor.into(), dim))
    }

    fn min(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_min(tensor.into()))
    }

    fn min_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_min_dim(tensor.into(), dim))
    }

    fn min_dim_with_indices(tensor: BridgeTensor, dim: usize) -> (BridgeTensor, BridgeTensor) {
        let (values, indices) = Dispatch::int_min_dim_with_indices(tensor.into(), dim);
        (BridgeTensor::int(values), BridgeTensor::int(indices))
    }

    fn clamp(tensor: BridgeTensor, min: Scalar, max: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_clamp(tensor.into(), min, max))
    }

    fn clamp_min(tensor: BridgeTensor, min: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_clamp_min(tensor.into(), min))
    }

    fn clamp_max(tensor: BridgeTensor, max: Scalar) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_clamp_max(tensor.into(), max))
    }
}

impl BasicAutodiffOps for Int {
    fn inner(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_inner(tensor.into()))
    }

    fn from_inner(inner: BridgeTensor) -> BridgeTensor {
        BridgeTensor::int(Dispatch::int_from_inner(inner.into()))
    }
}
