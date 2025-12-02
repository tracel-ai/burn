use alloc::vec::Vec;
use burn_std::{DType, Shape, Slice};

use crate::{
    Backend, ExecutionError, TensorData,
    element::{Element, ElementConversion},
    ops::TransactionPrimitive,
    tensor::{BasicOps, Bool, Device, IndexingUpdateOp, IntTensor},
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

    fn full<E: ElementConversion>(
        shape: Shape,
        fill_value: E,
        device: &Device<B>,
        dtype: DType,
    ) -> Self::Primitive {
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
            IndexingUpdateOp::Add => B::bool_select_add(tensor, dim, indices, values),
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

    fn from_data_dtype(data: TensorData, device: &Device<B>, dtype: DType) -> Self::Primitive {
        // Backends only use one bool representation dtype
        if dtype != B::BoolElem::dtype() {
            panic!("Expected bool dtype, got {dtype:?}")
        }
        B::bool_from_data(data.convert_dtype(dtype), device)
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
