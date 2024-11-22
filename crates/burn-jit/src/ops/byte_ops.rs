use super::{expand, numeric, permute};
use crate::kernel::{launch_unary, unary_op, UnaryOp};
use crate::{
    element::{BoolElement, ByteElement},
    kernel::prng::{random_bernoulli, random_normal, random_uniform},
};
use crate::{kernel, FloatElement, IntElement, JitBackend, JitRuntime};
use burn_tensor::ops::{
    BoolTensor, ByteElem, ByteTensor, ByteTensorOps, Device, FloatTensor, IntTensor,
};
use burn_tensor::{Distribution, ElementConversion, Shape, TensorData};
use cubecl::frontend::Numeric;
use cubecl::prelude::*;
use std::ops::Range;

impl<R, F, I, B, P> ByteTensorOps<Self> for JitBackend<R, F, I, B, P>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    B: BoolElement,
    P: ByteElement,
{
    fn byte_empty(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        super::empty::<R, P>(shape, device)
    }

    async fn byte_into_data(tensor: ByteTensor<Self>) -> TensorData {
        super::into_data::<R, P>(tensor).await
    }

    fn byte_from_data(data: TensorData, device: &Device<Self>) -> ByteTensor<Self> {
        super::from_data::<R, P>(data, device)
    }

    fn byte_device(tensor: &ByteTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn byte_to_device(tensor: ByteTensor<Self>, device: &Device<Self>) -> ByteTensor<Self> {
        super::to_device(tensor, device)
    }

    fn byte_reshape(tensor: ByteTensor<Self>, shape: Shape) -> ByteTensor<Self> {
        super::reshape(tensor, shape)
    }

    fn byte_slice(tensor: ByteTensor<Self>, ranges: &[Range<usize>]) -> ByteTensor<Self> {
        kernel::slice::<R, P>(tensor, ranges)
    }

    fn byte_slice_assign(
        tensor: ByteTensor<Self>,
        ranges: &[Range<usize>],
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        kernel::slice_assign::<R, P>(tensor, ranges, value)
    }

    fn byte_mask_where(
        tensor: ByteTensor<Self>,
        mask: BoolTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        kernel::mask_where_auto::<R, P, B>(tensor, mask, value)
    }

    fn byte_mask_fill(
        tensor: ByteTensor<Self>,
        mask: BoolTensor<Self>,
        value: ByteElem<Self>,
    ) -> ByteTensor<Self> {
        kernel::mask_fill_auto::<R, P, B>(tensor, mask, value)
    }

    fn byte_gather(
        dim: usize,
        tensor: ByteTensor<Self>,
        indices: IntTensor<Self>,
    ) -> ByteTensor<Self> {
        kernel::gather::<R, P, I>(dim, tensor, indices)
    }

    fn byte_scatter(
        dim: usize,
        tensor: ByteTensor<Self>,
        indices: IntTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        kernel::scatter::<R, P, I>(dim, tensor, indices, value)
    }

    fn byte_select(
        tensor: ByteTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> ByteTensor<Self> {
        kernel::select::<R, P, I>(tensor, dim, indices)
    }

    fn byte_select_assign(
        tensor: ByteTensor<Self>,
        dim: usize,
        indices: ByteTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        kernel::select_assign::<R, P, I>(tensor, dim, indices, value)
    }

    fn byte_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        kernel::equal::<R, P, B>(lhs, rhs)
    }

    fn byte_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        kernel::equal_elem::<R, P, B>(lhs, rhs)
    }

    fn byte_greater(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        kernel::greater::<R, P, B>(lhs, rhs)
    }

    fn byte_greater_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        kernel::greater_elem::<R, P, B>(lhs, rhs)
    }

    fn byte_greater_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        kernel::greater_equal::<R, P, B>(lhs, rhs)
    }

    fn byte_greater_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        kernel::greater_equal_elem::<R, P, B>(lhs, rhs)
    }

    fn byte_lower(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        kernel::lower::<R, P, B>(lhs, rhs)
    }

    fn byte_lower_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        kernel::lower_elem::<R, P, B>(lhs, rhs)
    }

    fn byte_lower_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        kernel::lower_equal::<R, P, B>(lhs, rhs)
    }

    fn byte_lower_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        kernel::lower_equal_elem::<R, P, B>(lhs, rhs)
    }

    fn byte_add(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        numeric::add::<R, P>(lhs, rhs)
    }

    fn byte_add_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        numeric::add_scalar::<R, P>(lhs, rhs)
    }

    fn byte_sub(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        numeric::sub::<R, P>(lhs, rhs)
    }

    fn byte_sub_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        numeric::sub_scalar::<R, P>(lhs, rhs)
    }

    fn byte_mul(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        numeric::mul::<R, P>(lhs, rhs)
    }

    fn byte_mul_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        numeric::mul_scalar::<R, P>(lhs, rhs)
    }

    fn byte_div(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        numeric::div::<R, P>(lhs, rhs)
    }

    fn byte_div_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        numeric::div_scalar::<R, P>(lhs, rhs)
    }

    fn byte_remainder(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        numeric::remainder::<R, P>(lhs, rhs)
    }

    fn byte_remainder_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        numeric::remainder_scalar::<R, P>(lhs, rhs)
    }

    fn byte_zeros(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        numeric::zeros::<R, P>(shape, device)
    }

    fn byte_ones(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        numeric::ones::<R, P>(shape, device)
    }

    fn byte_sum(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        kernel::reduce::sum::<R, P>(tensor, Default::default())
    }

    fn byte_sum_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        kernel::reduce::sum_dim::<R, P, I>(tensor, dim, Default::default())
    }

    fn byte_prod(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        kernel::reduce::prod::<R, P>(tensor, Default::default())
    }

    fn byte_prod_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        kernel::reduce::prod_dim::<R, P, I>(tensor, dim, Default::default())
    }

    fn byte_mean_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        kernel::reduce::mean_dim::<R, P, I>(tensor, dim, Default::default())
    }

    fn byte_argmax(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        kernel::reduce::argmax::<R, P, I>(tensor, dim, Default::default())
    }

    fn byte_argmin(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        kernel::reduce::argmin::<R, P, I>(tensor, dim, Default::default())
    }

    fn byte_clamp(
        tensor: ByteTensor<Self>,
        min: ByteElem<Self>,
        max: ByteElem<Self>,
    ) -> ByteTensor<Self> {
        kernel::clamp::<R, P>(tensor, min, max)
    }

    fn byte_abs(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        unary_op!(int(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Numeric>(input: Line<C>) -> Line<C> {
                Line::abs(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn byte_into_float(tensor: ByteTensor<Self>) -> FloatTensor<Self> {
        kernel::cast::<R, P, F>(tensor)
    }

    fn byte_into_int(tensor: ByteTensor<Self>) -> IntTensor<Self> {
        kernel::cast::<R, P, I>(tensor)
    }

    fn byte_swap_dims(mut tensor: ByteTensor<Self>, dim1: usize, dim2: usize) -> ByteTensor<Self> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    fn byte_repeat_dim(tensor: ByteTensor<Self>, dim: usize, times: usize) -> ByteTensor<Self> {
        kernel::repeat_dim::<R, P>(tensor, dim, times)
    }

    fn byte_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> ByteTensor<Self> {
        let float_tensor = match distribution {
            Distribution::Default => random_uniform(shape, device, 0.elem::<F>(), 255.elem()),
            Distribution::Uniform(low, high) => {
                random_uniform(shape, device, low.elem::<F>(), high.elem())
            }
            Distribution::Bernoulli(prob) => random_bernoulli(shape, device, prob.elem::<F>()),
            Distribution::Normal(mean, std) => {
                random_normal(shape, device, mean.elem::<F>(), std.elem())
            }
        };

        kernel::cast::<R, F, I>(float_tensor)
    }

    fn byte_permute(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        permute(tensor, axes)
    }

    fn byte_expand(tensor: ByteTensor<Self>, shape: Shape) -> ByteTensor<Self> {
        expand(tensor, shape)
    }

    fn byte_flip(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        kernel::flip::<R, P, B>(tensor, axes)
    }
}
