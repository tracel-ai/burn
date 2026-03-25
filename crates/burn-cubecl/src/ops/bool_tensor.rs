use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    element::BoolElement,
    kernel::{self, AndOp, OrOp},
};
use burn_backend::{
    ExecutionError, Slice,
    ops::BoolTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_backend::{Scalar, Shape, TensorData};
use burn_std::{BoolDType, BoolStore, DType, FloatDType, IntDType};
use cubecl::prelude::InputScalar;
use std::ops::Range;

use super::{expand, numeric, permute, unfold};

impl<R, F, I, BT> BoolTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn bool_empty(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        super::empty(shape, device, dtype.into())
    }

    fn bool_zeros(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        numeric::zeros(device.clone(), shape, dtype.into())
    }

    fn bool_ones(shape: Shape, device: &Device<Self>, dtype: BoolDType) -> BoolTensor<Self> {
        numeric::ones(device.clone(), shape, dtype.into())
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        super::into_data(tensor).await
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        if !matches!(
            data.dtype,
            DType::Bool(BoolStore::U8) | DType::Bool(BoolStore::U32)
        ) {
            unimplemented!("Unsupported dtype for `bool_from_data` {:?}", data.dtype);
        }
        super::from_data(data, device)
    }

    fn bool_into_int(tensor: BoolTensor<Self>, out_dtype: IntDType) -> IntTensor<Self> {
        kernel::bool_cast(tensor, out_dtype.into())
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self> {
        super::to_device(tensor, device)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        super::reshape(tensor, shape)
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        // Check if all steps are 1
        let all_steps_one = slices.iter().all(|info| info.step == 1);

        if all_steps_one {
            // Use optimized slice for step=1
            let simple_ranges: Vec<Range<usize>> = slices
                .iter()
                .enumerate()
                .map(|(i, slice)| slice.to_range(tensor.meta.shape()[i]))
                .collect();

            kernel::slice(tensor, &simple_ranges)
        } else {
            // Use slice with steps kernel
            kernel::slice_with_steps(tensor, slices)
        }
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        ranges: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::equal(lhs, rhs, dtype)
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let dtype = tensor.dtype;
        let scalar = match dtype {
            DType::Bool(BoolStore::U32) => InputScalar::new(u32::false_val(), dtype),
            DType::Bool(BoolStore::U8) => InputScalar::new(u8::false_val(), dtype),
            other => unimplemented!("Unsupported dtype for `bool_from_data` {other:?}"),
        };
        kernel::equal_elem(tensor, scalar, dtype)
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::launch_binop::<R, AndOp>(lhs, rhs)
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        kernel::launch_binop::<R, OrOp>(lhs, rhs)
    }

    fn bool_into_float(tensor: BoolTensor<Self>, out_dtype: FloatDType) -> FloatTensor<Self> {
        kernel::bool_cast(tensor, out_dtype.into())
    }

    fn bool_swap_dims(mut tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        tensor.meta.swap(dim1, dim2);

        tensor
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        kernel::repeat_dim(tensor, dim, times)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        permute(tensor, axes)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        expand(tensor, shape)
    }

    fn bool_select(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        kernel::select(tensor, dim, indices)
    }

    fn bool_select_or(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        kernel::select_assign(tensor, dim, indices, value, true)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let dtype = tensor.dtype;
        kernel::flip(tensor, axes, dtype)
    }

    fn bool_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        unfold(tensor, dim, size, step)
    }

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let dtype = tensor.dtype;
        kernel::mask_where_auto(tensor, mask, value, dtype)
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        let dtype = tensor.dtype;
        kernel::mask_fill_auto(tensor, mask, InputScalar::new(value, dtype), dtype)
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        kernel::gather(dim, tensor, indices)
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        kernel::scatter(dim, tensor, indices, value, true)
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::equal_elem(lhs, InputScalar::new(rhs, dtype), dtype)
    }
}
