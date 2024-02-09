use super::numeric;
use crate::codegen::dialect::gpu::{Elem, Item, Operation, UnaryOperation, Variable};
use crate::codegen::Compiler;
use crate::kernel::reduce::{self, init_reduce_output};
use crate::{kernel, unary, GpuBackend, JitRuntime};
use burn_tensor::ops::{BoolTensor, Device, FloatTensor, IntElem, IntTensor};

use burn_tensor::Reader;
use burn_tensor::{ops::IntTensorOps, Data, Shape};
use std::ops::Range;

impl<B: JitRuntime> IntTensorOps<Self> for GpuBackend<B> {
    fn int_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        super::empty::<B, _, D>(shape, device)
    }

    fn int_shape<const D: usize>(tensor: &IntTensor<Self, D>) -> Shape<D> {
        tensor.shape.clone()
    }

    fn int_into_data<const D: usize>(tensor: IntTensor<Self, D>) -> Reader<Data<IntElem<Self>, D>> {
        super::into_data(tensor)
    }

    fn int_from_data<const D: usize>(
        data: Data<IntElem<Self>, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        super::from_data::<B, _, D>(data, device)
    }

    fn int_device<const D: usize>(tensor: &IntTensor<Self, D>) -> Device<Self> {
        tensor.device.clone()
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<Self, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        super::to_device::<B, _, D>(tensor, device)
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        super::reshape(tensor, shape)
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> IntTensor<Self, D1> {
        kernel::slice(tensor, ranges)
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: IntTensor<Self, D1>,
    ) -> IntTensor<Self, D1> {
        kernel::slice_assign::<B, _, D1, D2>(tensor, ranges, value)
    }

    fn int_mask_where<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        kernel::mask_where(tensor, mask, value)
    }

    fn int_mask_fill<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        kernel::mask_fill(tensor, mask, value)
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        kernel::gather(dim, tensor, indices)
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        kernel::scatter::<B, _, _, D>(dim, tensor, indices, value)
    }

    fn int_select<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> IntTensor<Self, D> {
        kernel::select(tensor, dim, indices)
    }

    fn int_select_assign<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        kernel::select_assign::<B, _, _, D>(tensor, dim, indices, value)
    }

    fn int_cat<const D: usize>(tensors: Vec<IntTensor<Self, D>>, dim: usize) -> IntTensor<Self, D> {
        kernel::cat(tensors, dim)
    }

    fn int_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::equal::<B, _, D>(lhs, rhs)
    }

    fn int_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::equal_elem::<B, _, D>(lhs, rhs)
    }

    fn int_greater<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater::<B, _, D>(lhs, rhs)
    }

    fn int_greater_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_elem::<B, _, D>(lhs, rhs)
    }

    fn int_greater_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal::<B, _, D>(lhs, rhs)
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal_elem::<B, _, D>(lhs, rhs)
    }

    fn int_lower<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower::<B, _, D>(lhs, rhs)
    }

    fn int_lower_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_elem::<B, _, D>(lhs, rhs)
    }

    fn int_lower_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal::<B, _, D>(lhs, rhs)
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal_elem::<B, _, D>(lhs, rhs)
    }

    fn int_add<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        numeric::add::<B, _, D>(lhs, rhs)
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        numeric::add_scalar::<B, _, D>(lhs, rhs)
    }

    fn int_sub<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        numeric::sub::<B, _, D>(lhs, rhs)
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        numeric::sub_scalar::<B, _, D>(lhs, rhs)
    }

    fn int_mul<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        numeric::mul::<B, _, D>(lhs, rhs)
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        numeric::mul_scalar::<B, _, D>(lhs, rhs)
    }

    fn int_div<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        numeric::div::<B, _, D>(lhs, rhs)
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        numeric::div_scalar::<B, _, D>(lhs, rhs)
    }

    fn int_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        numeric::zeros::<B, _, D>(shape, device)
    }

    fn int_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        numeric::ones::<B, _, D>(shape, device)
    }

    fn int_sum<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        kernel::reduce::sum(tensor)
    }

    fn int_sum_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        let output = init_reduce_output(&tensor, dim);
        reduce::sum_dim(tensor, output, dim)
    }

    fn int_mean_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        let output = init_reduce_output(&tensor, dim);
        reduce::mean_dim(tensor, output, dim)
    }

    fn int_argmax<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        kernel::reduce::argmax(tensor, dim)
    }

    fn int_argmin<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        kernel::reduce::argmin(tensor, dim)
    }

    fn int_clamp<const D: usize>(
        tensor: IntTensor<Self, D>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        kernel::clamp::<B, _, D>(tensor, min, max)
    }

    fn int_abs<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, D> {
        unary!(
            operation: |elem: Elem| Operation::Abs(UnaryOperation {
                input: Variable::Input(0, Item::Scalar(elem)),
                out: Variable::Local(0, Item::Scalar(elem)),
            }),
            backend: B,
            input: tensor,
            elem: IntElem<Self>
        )
    }

    fn int_into_float<const D: usize>(tensor: IntTensor<Self, D>) -> FloatTensor<Self, D> {
        kernel::cast(tensor)
    }

    fn int_swap_dims<const D: usize>(
        mut tensor: IntTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<Self, D> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    fn int_repeat<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> IntTensor<Self, D> {
        kernel::repeat(tensor, dim, times)
    }
}
