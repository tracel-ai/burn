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
    ops::BridgeTensor,
};

impl TransactionOp for Bool {
    fn register_transaction(tr: &mut TransactionPrimitive<Dispatch>, tensor: BridgeTensor) {
        tr.register_bool(tensor.into());
    }
}

impl BasicOps for Bool {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        BridgeTensor::Bool(Dispatch::bool_empty(shape, &device.dispatch, dtype.into()))
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        BridgeTensor::Bool(Dispatch::bool_zeros(shape, &device.dispatch, dtype.into()))
    }
    fn ones(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        BridgeTensor::Bool(Dispatch::bool_ones(shape, &device.dispatch, dtype.into()))
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> BridgeTensor {
        if !dtype.is_bool() {
            panic!("Expected bool data type, got {dtype:?}");
        }
        if fill_value.elem() {
            BridgeTensor::Bool(Dispatch::bool_ones(shape, &device.dispatch, dtype.into()))
        } else {
            BridgeTensor::Bool(Dispatch::bool_zeros(shape, &device.dispatch, dtype.into()))
        }
    }

    fn reshape(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_reshape(tensor.into(), shape))
    }

    fn transpose(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_transpose(tensor.into()))
    }

    fn swap_dims(tensor: BridgeTensor, dim1: usize, dim2: usize) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_swap_dims(tensor.into(), dim1, dim2))
    }

    fn slice(tensor: BridgeTensor, slices: &[Slice]) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_slice(tensor.into(), slices))
    }

    fn slice_assign(tensor: BridgeTensor, slices: &[Slice], value: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_slice_assign(
            tensor.into(),
            slices,
            value.into(),
        ))
    }

    fn select(tensor: BridgeTensor, dim: usize, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_select(tensor.into(), dim, indices.into()))
    }

    fn select_assign(
        tensor: BridgeTensor,
        dim: usize,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        match update {
            IndexingUpdateOp::Add => BridgeTensor::Bool(Dispatch::bool_select_or(
                tensor.into(),
                dim,
                indices.into(),
                values.into(),
            )),
            _ => unimplemented!(),
        }
    }

    fn mask_where(tensor: BridgeTensor, mask: BridgeTensor, source: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_mask_where(
            tensor.into(),
            mask.into(),
            source.into(),
        ))
    }

    fn mask_fill(tensor: BridgeTensor, mask: BridgeTensor, value: Scalar) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_mask_fill(tensor.into(), mask.into(), value))
    }

    fn gather(dim: usize, tensor: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_gather(dim, tensor.into(), indices.into()))
    }

    fn scatter(
        dim: usize,
        tensor: BridgeTensor,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        match update {
            IndexingUpdateOp::Add => BridgeTensor::Bool(Dispatch::bool_scatter_or(
                dim,
                tensor.into(),
                indices.into(),
                values.into(),
            )),
            _ => unimplemented!(),
        }
    }

    fn scatter_nd(
        _data: BridgeTensor,
        _indices: BridgeTensor,
        _values: BridgeTensor,
        _reduction: IndexingUpdateOp,
    ) -> BridgeTensor {
        panic!("scatter_nd is not supported for bool tensors")
    }

    fn gather_nd(_data: BridgeTensor, _indices: BridgeTensor) -> BridgeTensor {
        panic!("gather_nd is not supported for bool tensors")
    }

    fn device(tensor: &BridgeTensor) -> Device {
        Dispatch::bool_device(tensor.as_dispatch()).into()
    }

    fn to_device(tensor: BridgeTensor, device: &Device) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_to_device(tensor.into(), &device.dispatch))
    }

    async fn into_data_async(tensor: BridgeTensor) -> Result<TensorData, ExecutionError> {
        Dispatch::bool_into_data(tensor.into()).await
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_from_data(
            data.convert_dtype(dtype),
            &device.dispatch,
        ))
    }

    fn repeat_dim(tensor: BridgeTensor, dim: usize, times: usize) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_repeat_dim(tensor.into(), dim, times))
    }

    fn equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_equal(lhs.into(), rhs.into()))
    }

    fn not_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_not_equal(lhs.into(), rhs.into()))
    }

    fn equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_equal_elem(lhs.into(), rhs))
    }

    fn not_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_not_equal_elem(lhs.into(), rhs))
    }

    fn cat(vectors: Vec<BridgeTensor>, dim: usize) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_cat(
            BridgeTensor::into_dispatch_vec(vectors),
            dim,
        ))
    }

    fn any(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_any(tensor.into()))
    }

    fn any_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_any_dim(tensor.into(), dim))
    }

    fn all(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_all(tensor.into()))
    }

    fn all_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_all_dim(tensor.into(), dim))
    }

    fn permute(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_permute(tensor.into(), axes))
    }

    fn expand(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_expand(tensor.into(), shape))
    }

    fn flip(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_flip(tensor.into(), axes))
    }

    fn unfold(tensor: BridgeTensor, dim: usize, size: usize, step: usize) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_unfold(tensor.into(), dim, size, step))
    }
}

impl BasicAutodiffOps for Bool {
    fn inner(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_inner(tensor.into()))
    }

    fn from_inner(inner: BridgeTensor) -> BridgeTensor {
        BridgeTensor::Bool(Dispatch::bool_from_inner(inner.into()))
    }
}
