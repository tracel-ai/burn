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
    ops::PrimitiveKind,
};

impl TransactionOp for Bool {
    fn register_transaction(tr: &mut TransactionPrimitive<Dispatch>, tensor: PrimitiveKind) {
        tr.register_bool(tensor.into());
    }
}

impl BasicOps for Bool {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> PrimitiveKind {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        PrimitiveKind::Bool(Dispatch::bool_empty(shape, &device.dispatch, dtype.into()))
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> PrimitiveKind {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        PrimitiveKind::Bool(Dispatch::bool_zeros(shape, &device.dispatch, dtype.into()))
    }
    fn ones(shape: Shape, device: &Device, dtype: DType) -> PrimitiveKind {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        PrimitiveKind::Bool(Dispatch::bool_ones(shape, &device.dispatch, dtype.into()))
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> PrimitiveKind {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        if fill_value.elem() {
            PrimitiveKind::Bool(Dispatch::bool_ones(shape, &device.dispatch, dtype.into()))
        } else {
            PrimitiveKind::Bool(Dispatch::bool_zeros(shape, &device.dispatch, dtype.into()))
        }
    }

    fn reshape(tensor: PrimitiveKind, shape: Shape) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_reshape(tensor.into(), shape))
    }

    fn transpose(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_transpose(tensor.into()))
    }

    fn swap_dims(tensor: PrimitiveKind, dim1: usize, dim2: usize) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_swap_dims(tensor.into(), dim1, dim2))
    }

    fn slice(tensor: PrimitiveKind, slices: &[Slice]) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_slice(tensor.into(), slices))
    }

    fn slice_assign(
        tensor: PrimitiveKind,
        slices: &[Slice],
        value: PrimitiveKind,
    ) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_slice_assign(
            tensor.into(),
            slices,
            value.into(),
        ))
    }

    fn select(tensor: PrimitiveKind, dim: usize, indices: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_select(tensor.into(), dim, indices.into()))
    }

    fn select_assign(
        tensor: PrimitiveKind,
        dim: usize,
        indices: PrimitiveKind,
        values: PrimitiveKind,
        update: IndexingUpdateOp,
    ) -> PrimitiveKind {
        match update {
            IndexingUpdateOp::Add => PrimitiveKind::Bool(Dispatch::bool_select_or(
                tensor.into(),
                dim,
                indices.into(),
                values.into(),
            )),
            _ => unimplemented!(),
        }
    }

    fn mask_where(
        tensor: PrimitiveKind,
        mask: PrimitiveKind,
        source: PrimitiveKind,
    ) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_mask_where(
            tensor.into(),
            mask.into(),
            source.into(),
        ))
    }

    fn mask_fill(tensor: PrimitiveKind, mask: PrimitiveKind, value: Scalar) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_mask_fill(tensor.into(), mask.into(), value))
    }

    fn gather(dim: usize, tensor: PrimitiveKind, indices: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_gather(dim, tensor.into(), indices.into()))
    }

    fn scatter(
        dim: usize,
        tensor: PrimitiveKind,
        indices: PrimitiveKind,
        values: PrimitiveKind,
        update: IndexingUpdateOp,
    ) -> PrimitiveKind {
        match update {
            IndexingUpdateOp::Add => PrimitiveKind::Bool(Dispatch::bool_scatter_or(
                dim,
                tensor.into(),
                indices.into(),
                values.into(),
            )),
            _ => unimplemented!(),
        }
    }

    fn scatter_nd(
        _data: PrimitiveKind,
        _indices: PrimitiveKind,
        _values: PrimitiveKind,
        _reduction: IndexingUpdateOp,
    ) -> PrimitiveKind {
        panic!("scatter_nd is not supported for bool tensors")
    }

    fn gather_nd(_data: PrimitiveKind, _indices: PrimitiveKind) -> PrimitiveKind {
        panic!("gather_nd is not supported for bool tensors")
    }

    fn device(tensor: &PrimitiveKind) -> Device {
        Dispatch::bool_device(tensor.as_dispatch()).into()
    }

    fn to_device(tensor: PrimitiveKind, device: &Device) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_to_device(tensor.into(), &device.dispatch))
    }

    async fn into_data_async(tensor: PrimitiveKind) -> Result<TensorData, ExecutionError> {
        Dispatch::bool_into_data(tensor.into()).await
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_from_data(
            data.convert_dtype(dtype),
            &device.dispatch,
        ))
    }

    fn repeat_dim(tensor: PrimitiveKind, dim: usize, times: usize) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_repeat_dim(tensor.into(), dim, times))
    }

    fn equal(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_equal(lhs.into(), rhs.into()))
    }

    fn not_equal(lhs: PrimitiveKind, rhs: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_not_equal(lhs.into(), rhs.into()))
    }

    fn equal_elem(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_equal_elem(lhs.into(), rhs))
    }

    fn not_equal_elem(lhs: PrimitiveKind, rhs: Scalar) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_not_equal_elem(lhs.into(), rhs))
    }

    fn cat(vectors: Vec<PrimitiveKind>, dim: usize) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_cat(
            PrimitiveKind::into_dispatch_vec(vectors),
            dim,
        ))
    }

    fn any(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_any(tensor.into()))
    }

    fn any_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_any_dim(tensor.into(), dim))
    }

    fn all(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_all(tensor.into()))
    }

    fn all_dim(tensor: PrimitiveKind, dim: usize) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_all_dim(tensor.into(), dim))
    }

    fn permute(tensor: PrimitiveKind, axes: &[usize]) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_permute(tensor.into(), axes))
    }

    fn expand(tensor: PrimitiveKind, shape: Shape) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_expand(tensor.into(), shape))
    }

    fn flip(tensor: PrimitiveKind, axes: &[usize]) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_flip(tensor.into(), axes))
    }

    fn unfold(tensor: PrimitiveKind, dim: usize, size: usize, step: usize) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_unfold(tensor.into(), dim, size, step))
    }
}

impl BasicAutodiffOps for Bool {
    fn inner(tensor: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_inner(tensor.into()))
    }

    fn from_inner(inner: PrimitiveKind) -> PrimitiveKind {
        PrimitiveKind::Bool(Dispatch::bool_from_inner(inner.into()))
    }
}
