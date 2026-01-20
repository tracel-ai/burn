use alloc::vec::Vec;
use burn_std::{DType, Shape, Slice};

use crate::{
    AutodiffBackend, Backend, Distribution, ExecutionError, Scalar, TensorData,
    ops::TransactionPrimitive,
    tensor::{
        BasicAutodiffOps, BasicOps, BoolTensor, Device, IndexingUpdateOp, Int, IntTensor, Numeric,
        Ordered, TensorKind,
    },
};

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

    fn register_transaction(tr: &mut TransactionPrimitive<B>, tensor: Self::Primitive) {
        tr.register_int(tensor);
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
        }
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

    fn from_data(data: TensorData, device: &Device<B>) -> Self::Primitive {
        B::int_from_data(data.convert::<B::IntElem>(), device)
    }

    fn from_data_dtype(data: TensorData, device: &Device<B>, dtype: DType) -> Self::Primitive {
        if !dtype.is_int() {
            panic!("Expected int dtype, got {dtype:?}")
        }

        B::int_from_data(data.convert_dtype(dtype), device)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        B::int_repeat_dim(tensor, dim, times)
    }

    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor<B> {
        B::int_equal(lhs, rhs)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor<B> {
        B::int_not_equal(lhs, rhs)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_equal_elem(lhs, rhs)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_not_equal_elem(lhs, rhs)
    }

    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        B::int_cat(vectors, dim)
    }

    fn any(tensor: Self::Primitive) -> BoolTensor<B> {
        B::int_any(tensor)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor<B> {
        B::int_any_dim(tensor, dim)
    }

    fn all(tensor: Self::Primitive) -> BoolTensor<B> {
        B::int_all(tensor)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor<B> {
        B::int_all_dim(tensor, dim)
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

    fn powf(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_powf(lhs, B::int_into_float(rhs))
    }

    fn powf_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::int_powf_scalar(lhs, rhs)
    }

    fn powi(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        B::int_powi(lhs, rhs)
    }

    fn powi_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        B::int_powi_scalar(lhs, rhs)
    }

    fn random(shape: Shape, distribution: Distribution, device: &Device<B>) -> Self::Primitive {
        B::int_random(shape, distribution, device)
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        B::int_sign(tensor)
    }

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

impl<B: Backend> Ordered<B> for Int
where
    <B as crate::backend::Backend>::IntElem: crate::element::ElementComparison,
{
    fn cummin(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_cummin(tensor, dim)
    }

    fn cummax(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        B::int_cummax(tensor, dim)
    }

    fn greater(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_greater(lhs, rhs)
    }

    fn greater_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_greater_elem(lhs, rhs)
    }

    fn greater_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_greater_equal(lhs, rhs)
    }

    fn greater_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_greater_equal_elem(lhs, rhs)
    }

    fn lower(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_lower(lhs, rhs)
    }

    fn lower_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_lower_elem(lhs, rhs)
    }

    fn lower_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::int_lower_equal(lhs, rhs)
    }

    fn lower_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::int_lower_equal_elem(lhs, rhs)
    }

    fn argmax(tensor: Self::Primitive, dim: usize) -> IntTensor<B> {
        B::int_argmax(tensor, dim)
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
