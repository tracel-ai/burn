use super::{expand, numeric, permute};
use crate::kernel::prng::{random_bernoulli, random_normal, random_uniform};
use crate::kernel::{launch_unary, unary_op, UnaryOp};
use crate::{kernel, FloatElement, IntElement, JitBackend, JitRuntime};
use burn_tensor::ops::{BoolTensor, Device, FloatTensor, IntElem, IntTensor};
use burn_tensor::{ops::IntTensorOps, Distribution, ElementConversion, Shape, TensorData};
use cubecl::frontend::Numeric;
use cubecl::prelude::*;
use std::ops::Range;

impl<R, F, I> IntTensorOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    fn int_empty(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        super::empty(shape, device)
    }

    fn int_shape(tensor: &IntTensor<Self>) -> Shape {
        tensor.shape.clone()
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> TensorData {
        super::into_data(tensor).await
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        super::from_data(data, device)
    }

    fn int_device(tensor: &IntTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &Device<Self>) -> IntTensor<Self> {
        super::to_device(tensor, device)
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        super::reshape(tensor, shape)
    }

    fn int_slice(tensor: IntTensor<Self>, ranges: &[Range<usize>]) -> IntTensor<Self> {
        kernel::slice(tensor, ranges)
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        ranges: &[Range<usize>],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::mask_where_auto(tensor, mask, value)
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntElem<Self>,
    ) -> IntTensor<Self> {
        kernel::mask_fill_auto(tensor, mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::scatter(dim, tensor, indices, value)
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::select(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::select_assign(tensor, dim, indices, value)
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::equal(lhs, rhs)
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::equal_elem(lhs, rhs)
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::greater(lhs, rhs)
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::greater_elem(lhs, rhs)
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::greater_equal_elem(lhs, rhs)
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::lower(lhs, rhs)
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::lower_elem(lhs, rhs)
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::lower_equal_elem(lhs, rhs)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::add(lhs, rhs)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::add_scalar(lhs, rhs)
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::sub_scalar(lhs, rhs)
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::mul_scalar(lhs, rhs)
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::div(lhs, rhs)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::div_scalar(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::remainder_scalar(lhs, rhs)
    }

    fn int_zeros(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        numeric::zeros(shape, device)
    }

    fn int_ones(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        numeric::ones(shape, device)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        kernel::reduce::sum(tensor, Default::default())
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        kernel::reduce::sum_dim(tensor, dim, Default::default())
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        kernel::reduce::prod(tensor, Default::default())
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        kernel::reduce::prod_dim(tensor, dim, Default::default())
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        kernel::reduce::mean_dim(tensor, dim, Default::default())
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        kernel::reduce::argmax(tensor, dim, Default::default())
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        kernel::reduce::argmin(tensor, dim, Default::default())
    }

    fn int_clamp(
        tensor: IntTensor<Self>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self> {
        kernel::clamp(tensor, min, max)
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_op!(int(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Numeric>(input: C) -> C {
                C::abs(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        kernel::cast(tensor)
    }

    fn int_swap_dims(mut tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        kernel::repeat_dim(tensor, dim, times)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
        let float_tensor = match distribution {
            Distribution::Default => random_uniform(shape, device, 0.elem::<f32>(), 255.elem()),
            Distribution::Uniform(low, high) => {
                random_uniform(shape, device, low.elem(), high.elem())
            }
            Distribution::Bernoulli(prob) => random_bernoulli(shape, device, prob.elem()),
            Distribution::Normal(mean, std) => {
                random_normal(shape, device, mean.elem(), std.elem())
            }
        };

        kernel::cast(float_tensor)
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        permute(tensor, axes)
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        expand(tensor, shape)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        kernel::flip(tensor, axes)
    }
}
