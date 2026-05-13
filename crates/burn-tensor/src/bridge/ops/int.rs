use alloc::vec::Vec;
use burn_std::{DType, Shape, Slice};

use crate::{
    AutodiffBackend, Backend, Distribution, ExecutionError, Scalar, TensorData,
    get_device_settings,
    ops::TransactionPrimitive,
    tensor::{
        BasicAutodiffOps, BasicOps, BoolTensor, Device, IndexingUpdateOp, Int, IntTensor, Numeric,
        Ordered, TensorKind, TransactionOp,
    },
};

impl<B: Backend> TransactionOp<B> for Int {
    fn register_transaction(tr: &mut TransactionPrimitive<B>, tensor: Self::Primitive) {
        tr.register_int(tensor);
    }
}
impl<B: Backend> BasicOps<B> for Int {
    type Elem = B::IntElem;

    fn empty(shape: Shape, device: &Device<B>, dtype: DType) -> Self::Primitive {
        B::int_empty(shape, device, dtype.into())
    }

    fn zeros(shape: Shape, device: &Device<B>, dtype: DType) -> Self::Primitive {
        B::int_zeros(shape, device, dtype.into())
    }
    fn ones(shape: Shape, device: &Device<B>, dtype: DType) -> Self::Primitive {
        B::int_ones(shape, device, dtype.into())
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device<B>, dtype: DType) -> Self::Primitive {
        B::int_full(shape, fill_value, device, dtype.into())
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::int_reshape(tensor, shape)
    }

    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        B::int_transpose(tensor)
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        B::int_swap_dims(tensor, dim1, dim2)
    }

    fn slice(tensor: Self::Primitive, slices: &[Slice]) -> Self::Primitive {
        B::int_slice(tensor, slices)
    }

    fn slice_assign(
        tensor: Self::Primitive,
        slices: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive {
        B::int_slice_assign(tensor, slices, value)
    }

    fn select(tensor: Self::Primitive, dim: usize, indices: IntTensor<B>) -> Self::Primitive {
        B::int_select(tensor, dim, indices)
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: IntTensor<B>,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => B::int_select_add(tensor, dim, indices, values),
            _ => unimplemented!(),
        }
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        B::int_mask_where(tensor, mask, source)
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Scalar,
    ) -> Self::Primitive {
        B::int_mask_fill(tensor, mask, value)
    }

    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive {
        B::int_gather(dim, tensor, indices)
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => B::int_scatter_add(dim, tensor, indices, values),
            _ => unimplemented!(),
        }
    }

    fn scatter_nd(
        data: Self::Primitive,
        indices: IntTensor<B>,
        values: Self::Primitive,
        reduction: IndexingUpdateOp,
    ) -> Self::Primitive {
        B::int_scatter_nd(data, indices, values, reduction)
    }

    fn gather_nd(data: Self::Primitive, indices: IntTensor<B>) -> Self::Primitive {
        B::int_gather_nd(data, indices)
    }

    fn device(tensor: &Self::Primitive) -> Device<B> {
        B::int_device(tensor)
    }

    fn to_device(tensor: Self::Primitive, device: &Device<B>) -> Self::Primitive {
        B::int_to_device(tensor, device)
    }

    async fn into_data_async(tensor: Self::Primitive) -> Result<TensorData, ExecutionError> {
        B::int_into_data(tensor).await
    }

    fn from_data(data: TensorData, device: &Device<B>, dtype: DType) -> Self::Primitive {
        B::int_from_data(data.convert_dtype(dtype), device)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        B::int_repeat_dim(tensor, dim, times)
    }

    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor<B> {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_equal(lhs, rhs, out_dtype)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor<B> {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_not_equal(lhs, rhs, out_dtype)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_equal_elem(lhs, rhs, out_dtype)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_not_equal_elem(lhs, rhs, out_dtype)
    }

    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        B::int_cat(vectors, dim)
    }

    fn any(tensor: Self::Primitive) -> BoolTensor<B> {
        let out_dtype = get_device_settings::<B>(&B::int_device(&tensor)).bool_dtype;
        B::int_any(tensor, out_dtype)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor<B> {
        let out_dtype = get_device_settings::<B>(&B::int_device(&tensor)).bool_dtype;
        B::int_any_dim(tensor, dim, out_dtype)
    }

    fn all(tensor: Self::Primitive) -> BoolTensor<B> {
        let out_dtype = get_device_settings::<B>(&B::int_device(&tensor)).bool_dtype;
        B::int_all(tensor, out_dtype)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor<B> {
        let out_dtype = get_device_settings::<B>(&B::int_device(&tensor)).bool_dtype;
        B::int_all_dim(tensor, dim, out_dtype)
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::int_permute(tensor, axes)
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::int_expand(tensor, shape)
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::int_flip(tensor, axes)
    }

    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive {
        B::int_unfold(tensor, dim, size, step)
    }
}

impl<B: Backend> Numeric<B> for Int {
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_add(lhs, rhs)
    }
    fn add_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::int_add_scalar(lhs, rhs)
    }
    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_sub(lhs, rhs)
    }
    fn sub_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::int_sub_scalar(lhs, rhs)
    }
    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_div(lhs, rhs)
    }
    fn div_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::int_div_scalar(lhs, rhs)
    }
    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_remainder(lhs, rhs)
    }
    fn remainder_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::int_remainder_scalar(lhs, rhs)
    }
    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_mul(lhs, rhs)
    }
    fn mul_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::int_mul_scalar(lhs, rhs)
    }
    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        B::int_neg(tensor)
    }

    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        B::int_sum(tensor)
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_sum_dim(tensor, dim)
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        B::int_prod(tensor)
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_prod_dim(tensor, dim)
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        B::int_mean(tensor)
    }
    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_mean_dim(tensor, dim)
    }
    fn cumsum(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_cumsum(tensor, dim)
    }
    fn cumprod(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_cumprod(tensor, dim)
    }

    fn abs(tensor: Self::Primitive) -> Self::Primitive {
        B::int_abs(tensor)
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_powi(lhs, rhs)
    }

    fn powi_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::int_powi_scalar(lhs, rhs)
    }

    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<B>,
        dtype: DType,
    ) -> Self::Primitive {
        B::int_random(shape, distribution, device, dtype.into())
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        B::int_sign(tensor)
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    ///
    /// # Panics
    ///
    /// If the two tensors don't have a compatible shape.
    fn matmul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_matmul(lhs, rhs)
    }
}

impl<B: Backend> Ordered<B> for Int {
    fn sort(tensor: Self::Primitive, dim: usize, descending: bool) -> Self::Primitive {
        B::int_sort(tensor, dim, descending)
    }

    fn sort_with_indices(
        tensor: Self::Primitive,
        dim: usize,
        descending: bool,
    ) -> (Self::Primitive, IntTensor<B>) {
        B::int_sort_with_indices(tensor, dim, descending)
    }

    fn argsort(tensor: Self::Primitive, dim: usize, descending: bool) -> IntTensor<B> {
        B::int_argsort(tensor, dim, descending)
    }

    fn cummin(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_cummin(tensor, dim)
    }

    fn cummax(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_cummax(tensor, dim)
    }

    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_greater(lhs, rhs, out_dtype)
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_greater_elem(lhs, rhs, out_dtype)
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_greater_equal(lhs, rhs, out_dtype)
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_greater_equal_elem(lhs, rhs, out_dtype)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_lower(lhs, rhs, out_dtype)
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_lower_elem(lhs, rhs, out_dtype)
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_lower_equal(lhs, rhs, out_dtype)
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<B>(&B::int_device(&lhs)).bool_dtype;
        B::int_lower_equal_elem(lhs, rhs, out_dtype)
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        B::int_argmax(tensor, dim)
    }

    fn argtopk(tensor: Self::Primitive, dim: usize, k: usize) -> IntTensor<B> {
        B::int_argtopk(tensor, dim, k)
    }

    fn topk(tensor: Self::Primitive, dim: usize, k: usize) -> IntTensor<B> {
        B::int_topk(tensor, dim, k)
    }

    fn argmin(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        B::int_argmin(tensor, dim)
    }

    fn max(tensor: Self::Primitive) -> Self::Primitive {
        B::int_max(tensor)
    }

    fn max_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_max_dim(tensor, dim)
    }

    fn max_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, IntTensor<B>) {
        B::int_max_dim_with_indices(tensor, dim)
    }

    fn max_abs(tensor: Self::Primitive) -> Self::Primitive {
        B::int_max_abs(tensor)
    }

    fn max_abs_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_max_abs_dim(tensor, dim)
    }

    fn min(tensor: Self::Primitive) -> Self::Primitive {
        B::int_min(tensor)
    }

    fn min_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_min_dim(tensor, dim)
    }

    fn min_dim_with_indices(
        tensor: Self::Primitive,
        dim: usize,
    ) -> (Self::Primitive, IntTensor<B>) {
        B::int_min_dim_with_indices(tensor, dim)
    }

    fn clamp(tensor: Self::Primitive, min: Scalar, max: Scalar) -> Self::Primitive {
        B::int_clamp(tensor, min, max)
    }

    fn clamp_min(tensor: Self::Primitive, min: Scalar) -> Self::Primitive {
        B::int_clamp_min(tensor, min)
    }

    fn clamp_max(tensor: Self::Primitive, max: Scalar) -> Self::Primitive {
        B::int_clamp_max(tensor, max)
    }
}

impl<B: AutodiffBackend> BasicAutodiffOps<B> for Int {
    type InnerKind = Int;

    fn inner(
        tensor: <Self as TensorKind<B>>::Primitive,
    ) -> <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive {
        B::int_inner(tensor)
    }

    fn from_inner(
        inner: <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive,
    ) -> <Self as TensorKind<B>>::Primitive {
        B::int_from_inner(inner)
    }
}
