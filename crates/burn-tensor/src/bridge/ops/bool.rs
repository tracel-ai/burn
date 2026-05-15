use alloc::vec::Vec;
use burn_backend::{
    AutodiffBackend, Scalar, TensorData,
    ops::{BoolTensorOps, TransactionPrimitive},
};
use burn_dispatch::Dispatch;
use burn_std::{DType, ExecutionError, IndexingUpdateOp, Shape, Slice};

use crate::{
    Bool, Device,
    bridge::{BasicAutodiffOps, BasicOps, TransactionOp},
    ops::{BoolTensor, IntTensor},
};

impl TransactionOp for Bool {
    fn register_transaction(tr: &mut TransactionPrimitive<Dispatch>, tensor: Self::Primitive) {
        tr.register_bool(tensor);
    }
}

impl BasicOps for Bool {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> Self::Primitive {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        Dispatch::bool_empty(shape, &device.dispatch, dtype.into())
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> Self::Primitive {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        Dispatch::bool_zeros(shape, &device.dispatch, dtype.into())
    }
    fn ones(shape: Shape, device: &Device, dtype: DType) -> Self::Primitive {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        Dispatch::bool_ones(shape, &device.dispatch, dtype.into())
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> Self::Primitive {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        if fill_value.elem() {
            Dispatch::bool_ones(shape, &device.dispatch, dtype.into())
        } else {
            Dispatch::bool_zeros(shape, &device.dispatch, dtype.into())
        }
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        Dispatch::bool_reshape(tensor, shape)
    }

    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::bool_transpose(tensor)
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        Dispatch::bool_swap_dims(tensor, dim1, dim2)
    }

    fn slice(tensor: Self::Primitive, slices: &[Slice]) -> Self::Primitive {
        Dispatch::bool_slice(tensor, slices)
    }

    fn slice_assign(
        tensor: Self::Primitive,
        slices: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive {
        Dispatch::bool_slice_assign(tensor, slices, value)
    }

    fn select(tensor: Self::Primitive, dim: usize, indices: IntTensor) -> Self::Primitive {
        Dispatch::bool_select(tensor, dim, indices)
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: IntTensor,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => Dispatch::bool_select_or(tensor, dim, indices, values),
            _ => unimplemented!(),
        }
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: BoolTensor,
        source: Self::Primitive,
    ) -> Self::Primitive {
        Dispatch::bool_mask_where(tensor, mask, source)
    }

    fn mask_fill(tensor: Self::Primitive, mask: BoolTensor, value: Scalar) -> Self::Primitive {
        Dispatch::bool_mask_fill(tensor, mask, value)
    }

    fn gather(dim: usize, tensor: Self::Primitive, indices: IntTensor) -> Self::Primitive {
        Dispatch::bool_gather(dim, tensor, indices)
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: IntTensor,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => Dispatch::bool_scatter_or(dim, tensor, indices, values),
            _ => unimplemented!(),
        }
    }

    fn scatter_nd(
        _data: Self::Primitive,
        _indices: IntTensor,
        _values: Self::Primitive,
        _reduction: IndexingUpdateOp,
    ) -> Self::Primitive {
        panic!("scatter_nd is not supported for bool tensors")
    }

    fn gather_nd(_data: Self::Primitive, _indices: IntTensor) -> Self::Primitive {
        panic!("gather_nd is not supported for bool tensors")
    }

    fn device(tensor: &Self::Primitive) -> Device {
        Dispatch::bool_device(tensor).into()
    }

    fn to_device(tensor: Self::Primitive, device: &Device) -> Self::Primitive {
        Dispatch::bool_to_device(tensor, &device.dispatch)
    }

    async fn into_data_async(tensor: Self::Primitive) -> Result<TensorData, ExecutionError> {
        Dispatch::bool_into_data(tensor).await
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> Self::Primitive {
        Dispatch::bool_from_data(data.convert_dtype(dtype), &device.dispatch)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        Dispatch::bool_repeat_dim(tensor, dim, times)
    }

    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        Dispatch::bool_equal(lhs, rhs)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> BoolTensor {
        Dispatch::bool_not_equal(lhs, rhs)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        Dispatch::bool_equal_elem(lhs, rhs)
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> BoolTensor {
        Dispatch::bool_not_equal_elem(lhs, rhs)
    }

    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        Dispatch::bool_cat(vectors, dim)
    }

    fn any(tensor: Self::Primitive) -> BoolTensor {
        Dispatch::bool_any(tensor)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor {
        Dispatch::bool_any_dim(tensor, dim)
    }

    fn all(tensor: Self::Primitive) -> BoolTensor {
        Dispatch::bool_all(tensor)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> BoolTensor {
        Dispatch::bool_all_dim(tensor, dim)
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        Dispatch::bool_permute(tensor, axes)
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        Dispatch::bool_expand(tensor, shape)
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        Dispatch::bool_flip(tensor, axes)
    }

    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive {
        Dispatch::bool_unfold(tensor, dim, size, step)
    }
}

impl BasicAutodiffOps for Bool {
    fn inner(tensor: Self::Primitive) -> Self::Primitive {
        Dispatch::bool_inner(tensor)
    }

    fn from_inner(inner: Self::Primitive) -> Self::Primitive {
        Dispatch::bool_from_inner(inner)
    }
}
