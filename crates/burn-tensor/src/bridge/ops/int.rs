use alloc::vec::Vec;
use burn_backend::{
    AutodiffBackend, Distribution, Scalar, TensorData, get_device_settings,
    ops::{IntTensorOps, TransactionPrimitive},
};
use burn_dispatch::Dispatch;
use burn_std::{DType, ExecutionError, IndexingUpdateOp, Shape, Slice};

use crate::{
    Device, Int,
    bridge::{BasicAutodiffOps, BasicOps, Numeric, Ordered, TransactionOp},
    ops::{BoolTensor, IntTensor},
};

impl TransactionOp for Int {
    fn register_transaction(tr: &mut TransactionPrimitive<Dispatch>, tensor: Self::Primitive) {
        tr.register_int(tensor);
    }
}
impl BasicOps for Int {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> Self::Primitive {
        Dispatch::int_empty(shape, &device.dispatch, dtype.into())
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> Self::Primitive {
        Dispatch::int_zeros(shape, &device.dispatch, dtype.into())
    }
    fn ones(shape: Shape, device: &Device, dtype: DType) -> Self::Primitive {
        Dispatch::int_ones(shape, &device.dispatch, dtype.into())
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> Self::Primitive {
        Dispatch::int_full(shape, fill_value, &device.dispatch, dtype.into())
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        Dispatch::int_reshape(tensor, shape)
    }

    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_transpose(tensor)
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        Dispatch::int_swap_dims(tensor, dim1, dim2)
    }

    fn slice(tensor: Self::Primitive, slices: &[Slice]) -> Self::Primitive {
        Dispatch::int_slice(tensor, slices)
    }

    fn slice_assign(
        tensor: Self::Primitive,
        slices: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive {
        Dispatch::int_slice_assign(tensor, slices, value)
    }

    fn select(tensor: Self::Primitive, dim: usize, indices: IntTensor) -> Self::Primitive {
        Dispatch::int_select(tensor, dim, indices)
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: IntTensor,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => Dispatch::int_select_add(tensor, dim, indices, values),
            _ => unimplemented!(),
        }
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: BoolTensor,
        source: Self::Primitive,
    ) -> Self::Primitive {
        Dispatch::int_mask_where(tensor, mask, source)
    }

    fn mask_fill(tensor: Self::Primitive, mask: BoolTensor, value: Scalar) -> Self::Primitive {
        Dispatch::int_mask_fill(tensor, mask, value)
    }

    fn gather(dim: usize, tensor: Self::Primitive, indices: IntTensor) -> Self::Primitive {
        Dispatch::int_gather(dim, tensor, indices)
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: IntTensor,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => Dispatch::int_scatter_add(dim, tensor, indices, values),
            _ => unimplemented!(),
        }
    }

    fn scatter_nd(
        data: Self::Primitive,
        indices: IntTensor,
        values: Self::Primitive,
        reduction: IndexingUpdateOp,
    ) -> Self::Primitive {
        Dispatch::int_scatter_nd(data, indices, values, reduction)
    }

    fn gather_nd(data: Self::Primitive, indices: IntTensor) -> Self::Primitive {
        Dispatch::int_gather_nd(data, indices)
    }

    fn device(tensor: &Self::Primitive) -> Device {
        Dispatch::int_device(tensor).into()
    }

    fn to_device(tensor: Self::Primitive, device: &Device) -> Self::Primitive {
        Dispatch::int_to_device(tensor, &device.dispatch)
    }

    async fn into_data_async(tensor: Self::Primitive) -> Result<TensorData, ExecutionError> {
        Dispatch::int_into_data(tensor).await
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> Self::Primitive {
        Dispatch::int_from_data(data.convert_dtype(dtype), &device.dispatch)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        Dispatch::int_repeat_dim(tensor, dim, times)
    }

    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_equal(lhs, rhs, out_dtype)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_not_equal(lhs, rhs, out_dtype)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_equal_elem(lhs, rhs, out_dtype)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        Dispatch::int_cat(vectors, dim)
    }

    fn any(tensor: Self::Primitive) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&tensor)).bool_dtype;
        Dispatch::int_any(tensor, out_dtype)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&tensor)).bool_dtype;
        Dispatch::int_any_dim(tensor, dim, out_dtype)
    }

    fn all(tensor: Self::Primitive) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&tensor)).bool_dtype;
        Dispatch::int_all(tensor, out_dtype)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&tensor)).bool_dtype;
        Dispatch::int_all_dim(tensor, dim, out_dtype)
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        Dispatch::int_permute(tensor, axes)
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        Dispatch::int_expand(tensor, shape)
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        Dispatch::int_flip(tensor, axes)
    }

    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive {
        Dispatch::int_unfold(tensor, dim, size, step)
    }
}

impl Numeric for Int {
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        Dispatch::int_add(lhs, rhs)
    }
    fn add_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        Dispatch::int_add_scalar(lhs, rhs)
    }
    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        Dispatch::int_sub(lhs, rhs)
    }
    fn sub_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        Dispatch::int_sub_scalar(lhs, rhs)
    }
    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        Dispatch::int_div(lhs, rhs)
    }
    fn div_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        Dispatch::int_div_scalar(lhs, rhs)
    }
    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        Dispatch::int_remainder(lhs, rhs)
    }
    fn remainder_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        Dispatch::int_remainder_scalar(lhs, rhs)
    }
    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        Dispatch::int_mul(lhs, rhs)
    }
    fn mul_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        Dispatch::int_mul_scalar(lhs, rhs)
    }
    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_neg(tensor)
    }

    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_sum(tensor)
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_sum_dim(tensor, dim)
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_prod(tensor)
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_prod_dim(tensor, dim)
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_mean(tensor)
    }
    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_mean_dim(tensor, dim)
    }
    fn cumsum(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_cumsum(tensor, dim)
    }
    fn cumprod(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_cumprod(tensor, dim)
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_abs(tensor)
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        Dispatch::int_powi(lhs, rhs)
    }

    fn powi_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        Dispatch::int_powi_scalar(lhs, rhs)
    }

    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &Device,
        dtype: DType,
    ) -> Self::Primitive {
        Dispatch::int_random(shape, distribution, &device.dispatch, dtype.into())
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_sign(tensor)
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors don't have a compatible shape.
    fn matmul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        Dispatch::int_matmul(lhs, rhs)
    }
}

impl Ordered for Int {
    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive {
        Dispatch::int_sort(tensor, dim, descending)
    }

    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, IntTensor) {
        Dispatch::int_sort_with_indices(tensor, dim, descending)
    }

    fn argsort(tensor: Self::Primitive, dim: usize, descending: bool) -> IntTensor {
        Dispatch::int_argsort(tensor, dim, descending)
    }

    fn cummin(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_cummin(tensor, dim)
    }

    fn cummax(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_cummax(tensor, dim)
    }

    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_greater(lhs, rhs, out_dtype)
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_greater_elem(lhs, rhs, out_dtype)
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_greater_equal(lhs, rhs, out_dtype)
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_lower(lhs, rhs, out_dtype)
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_lower_elem(lhs, rhs, out_dtype)
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_lower_equal(lhs, rhs, out_dtype)
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        let out_dtype = get_device_settings::<Dispatch>(&Dispatch::int_device(&lhs)).bool_dtype;
        Dispatch::int_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> IntTensor {
        Dispatch::int_argmax(tensor, dim)
    }

    fn argtopk(tensor: Self::Primitive, dim: usize, k: usize) -> IntTensor {
        Dispatch::int_argtopk(tensor, dim, k)
    }

    fn topk(tensor: Self::Primitive, dim: usize, k: usize) -> IntTensor {
        Dispatch::int_topk(tensor, dim, k)
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> IntTensor {
        Dispatch::int_argmin(tensor, dim)
    }

    fn max(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_max(tensor)
    }

    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_max_dim(tensor, dim)
    }

    fn max_dim_with_indices(tensor: Self::Primitive, dim: usize) -> (Self::Primitive, IntTensor) {
        Dispatch::int_max_dim_with_indices(tensor, dim)
    }

    fn max_abs(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_max_abs(tensor)
    }

    fn max_abs_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_max_abs_dim(tensor, dim)
    }

    fn min(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_min(tensor)
    }

    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        Dispatch::int_min_dim(tensor, dim)
    }

    fn min_dim_with_indices(tensor: Self::Primitive, dim: usize) -> (Self::Primitive, IntTensor) {
        Dispatch::int_min_dim_with_indices(tensor, dim)
    }

    fn clamp(tensor: Self::Primitive, min: Scalar, max: Scalar) -> Self::Primitive {
        Dispatch::int_clamp(tensor, min, max)
    }

    fn clamp_min(tensor: Self::Primitive, min: Scalar) -> Self::Primitive {
        Dispatch::int_clamp_min(tensor, min)
    }

    fn clamp_max(tensor: Self::Primitive, max: Scalar) -> Self::Primitive {
        Dispatch::int_clamp_max(tensor, max)
    }
}

impl BasicAutodiffOps for Int {
    fn inner(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::int_inner(tensor)
    }

    fn from_inner(inner: Self::Primitive) -> Self::Primitive {
        Dispatch::int_from_inner(inner)
    }
}
