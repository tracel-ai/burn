use super::{expand, numeric, permute};
use crate::kernel::matmul::{matmul, MatmulStrategy};
use crate::kernel::prng::{random_bernoulli, random_normal, random_uniform};
use crate::kernel::{self, launch_unary, reduce, unary_op, UnaryOp};
use crate::JitBackend;
use crate::{FloatElement, IntElement, JitRuntime};
use burn_tensor::ops::{BoolTensor, Device, FloatElem, FloatTensor, IntTensor};
use burn_tensor::ElementConversion;
use burn_tensor::{ops::FloatTensorOps, Distribution, Shape, TensorData};
use cubecl::prelude::*;
use std::ops::Range;

impl<R, F, I> FloatTensorOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        super::from_data(data, device)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        match distribution {
            Distribution::Default => random_uniform(shape, device, 0.elem(), 1.elem()),
            Distribution::Uniform(low, high) => {
                random_uniform(shape, device, low.elem(), high.elem())
            }
            Distribution::Bernoulli(prob) => random_bernoulli(shape, device, prob.elem()),
            Distribution::Normal(mean, std) => {
                random_normal(shape, device, mean.elem(), std.elem())
            }
        }
    }

    fn float_shape(tensor: &FloatTensor<Self>) -> Shape {
        tensor.shape.clone()
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        super::into_data(tensor).await
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        super::to_device(tensor, device)
    }

    fn float_empty(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        super::empty(shape, device)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::add(lhs, rhs)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        numeric::add_scalar(lhs, rhs)
    }

    fn float_zeros(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        numeric::zeros(shape, device)
    }

    fn float_full(
        shape: Shape,
        fill_value: FloatElem<Self>,
        device: &R::Device,
    ) -> FloatTensor<Self> {
        numeric::full(shape, device, fill_value)
    }

    fn float_ones(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        numeric::ones(shape, device)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::sub(lhs, rhs)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        numeric::sub_scalar(lhs, rhs)
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::mul(lhs, rhs)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        numeric::mul_scalar(lhs, rhs)
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::div(lhs, rhs)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        numeric::div_scalar(lhs, rhs)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        numeric::remainder_scalar(lhs, rhs)
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        matmul(lhs, rhs, MatmulStrategy::default())
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        super::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        super::reshape(tensor, shape)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::gather(dim, tensor, indices)
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::scatter(dim, tensor, indices, value)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::select(tensor, dim, indices)
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::select_assign(tensor, dim, indices, value)
    }

    fn float_slice(tensor: FloatTensor<Self>, ranges: &[Range<usize>]) -> FloatTensor<Self> {
        kernel::slice(tensor, ranges)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[Range<usize>],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::mask_where_auto(tensor, mask, value)
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        kernel::mask_fill_auto(tensor, mask, value)
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::equal(lhs, rhs)
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        kernel::equal_elem(lhs, rhs)
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::greater(lhs, rhs)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        kernel::greater_elem(lhs, rhs)
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        kernel::greater_equal_elem(lhs, rhs)
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::lower(lhs, rhs)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        kernel::lower_elem(lhs, rhs)
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        kernel::lower_equal_elem(lhs, rhs)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::sum(tensor, Default::default())
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::sum_dim(tensor, dim, Default::default())
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::mean_dim(tensor, dim, Default::default())
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::prod(tensor, Default::default())
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::prod_dim(tensor, dim, Default::default())
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, input| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::exp(input)
            }
            execute::expand::<C>(context, input)
        })
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::log(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::log1p(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn float_powf_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        unary_op!(float(lhs, rhs.elem::<F>()) => |context, tensor, scalar| {
            #[cube]
            fn execute<C: Float>(input: C, scalar: C) -> C {
                C::powf(input, scalar)
            }
            execute::expand::<C>(context, tensor, scalar)
        })
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::sqrt(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::abs(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::cos(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::sin(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::tanh(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::erf(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::argmax(tensor, dim, Default::default())
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::argmin(tensor, dim, Default::default())
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        kernel::cast(tensor)
    }

    fn float_clamp(
        tensor: FloatTensor<Self>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        kernel::clamp(tensor, min, max)
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::recip(input)
            }
            execute::expand::<C>(context, tensor)
        })
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        kernel::repeat_dim(tensor, dim, times)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::pow(lhs, rhs)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        permute(tensor, axes)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        expand(tensor, shape)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        kernel::flip(tensor, axes)
    }
}
