use alloc::vec::Vec;
use burn_std::{DType, Shape, Slice};

use crate::{
    AutodiffBackend, Backend, ExecutionError, Scalar, TensorData,
    element::Element,
    ops::TransactionPrimitive,
    tensor::{BasicAutodiffOps, BasicOps, Bool, Device, IndexingUpdateOp, IntTensor, TensorKind},
};

impl<B: Backend> BasicOps<B> for Bool {
    type Elem = B::BoolElem;

    fn empty(shape: Shape, device: &Device<B>, dtype: DType) -> Self::Primitive {
        if dtype != Self::Elem::dtype() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        B::bool_empty(shape, device)
    }

    fn zeros(shape: Shape, device: &Device<B>, dtype: DType) -> Self::Primitive {
        if dtype != Self::Elem::dtype() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        B::bool_zeros(shape, device)
    }
    fn ones(shape: Shape, device: &Device<B>, dtype: DType) -> Self::Primitive {
        if dtype != Self::Elem::dtype() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        B::bool_ones(shape, device)
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device<B>, dtype: DType) -> Self::Primitive {
        if dtype != Self::Elem::dtype() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        if fill_value.elem() {
            B::bool_ones(shape, device)
        } else {
            B::bool_zeros(shape, device)
        }
    }

    fn register_transaction(tr: &mut TransactionPrimitive<B>, tensor: Self::Primitive) {
        tr.register_bool(tensor);
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::bool_reshape(tensor, shape)
    }

    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        B::bool_transpose(tensor)
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        B::bool_swap_dims(tensor, dim1, dim2)
    }

    fn slice(tensor: Self::Primitive, slices: &[Slice]) -> Self::Primitive {
        B::bool_slice(tensor, slices)
    }

    fn slice_assign(
        tensor: Self::Primitive,
        slices: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive {
        B::bool_slice_assign(tensor, slices, value)
    }

    fn select(tensor: Self::Primitive, dim: usize, indices: IntTensor<B>) -> Self::Primitive {
        B::bool_select(tensor, dim, indices)
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: IntTensor<B>,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => B::bool_select_or(tensor, dim, indices, values),
        }
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        B::bool_mask_where(tensor, mask, source)
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Scalar,
    ) -> Self::Primitive {
        B::bool_mask_fill(tensor, mask, value)
    }

    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive {
        B::bool_gather(dim, tensor, indices)
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => B::bool_scatter_or(dim, tensor, indices, values),
        }
    }

    fn device(tensor: &Self::Primitive) -> Device<B> {
        B::bool_device(tensor)
    }

    fn to_device(tensor: Self::Primitive, device: &Device<B>) -> Self::Primitive {
        B::bool_to_device(tensor, device)
    }

    async fn into_data_async(tensor: Self::Primitive) -> Result<TensorData, ExecutionError> {
        B::bool_into_data(tensor).await
    }

    fn from_data(data: TensorData, device: &Device<B>) -> Self::Primitive {
        B::bool_from_data(data.convert::<B::BoolElem>(), device)
    }

    fn from_data_dtype(data: TensorData, device: &Device<B>, _dtype: DType) -> Self::Primitive {
        // Bool tensors have exactly one representation per backend, so the
        // requested dtype is irrelevant. Convert to `B::BoolElem` directly.
        B::bool_from_data(data.convert::<B::BoolElem>(), device)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        B::bool_repeat_dim(tensor, dim, times)
    }

    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::bool_equal(lhs, rhs)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::bool_not_equal(lhs, rhs)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::bool_equal_elem(lhs, rhs)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive {
        B::bool_not_equal_elem(lhs, rhs)
    }

    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        B::bool_cat(vectors, dim)
    }

    fn any(tensor: Self::Primitive) -> B::BoolTensorPrimitive {
        B::bool_any(tensor)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> B::BoolTensorPrimitive {
        B::bool_any_dim(tensor, dim)
    }

    fn all(tensor: Self::Primitive) -> B::BoolTensorPrimitive {
        B::bool_all(tensor)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> B::BoolTensorPrimitive {
        B::bool_all_dim(tensor, dim)
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::bool_permute(tensor, axes)
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::bool_expand(tensor, shape)
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::bool_flip(tensor, axes)
    }

    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive {
        B::bool_unfold(tensor, dim, size, step)
    }
}

impl<B: AutodiffBackend> BasicAutodiffOps<B> for Bool {
    type InnerKind = Bool;

    fn inner(
        tensor: <Self as TensorKind<B>>::Primitive,
    ) -> <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive {
        B::bool_inner(tensor)
    }

    fn from_inner(
        inner: <Self::InnerKind as TensorKind<<B as AutodiffBackend>::InnerBackend>>::Primitive,
    ) -> <Self as TensorKind<B>>::Primitive {
        B::bool_from_inner(inner)
    }
}
