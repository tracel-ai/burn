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
    fn float_from_data<const D: usize>(
        data: TensorData,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        super::from_data(data, device)
    }

    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
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

    fn float_shape<const D: usize>(tensor: &FloatTensor<Self, D>) -> Shape<D> {
        tensor.shape.clone()
    }

    async fn float_into_data<const D: usize>(tensor: FloatTensor<Self, D>) -> TensorData {
        super::into_data(tensor).await
    }

    fn float_device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        tensor.device.clone()
    }

    fn float_to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        super::to_device(tensor, device)
    }

    fn float_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        super::empty(shape, device)
    }

    fn float_add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::add(lhs, rhs)
    }

    fn float_add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::add_scalar(lhs, rhs)
    }

    fn float_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        numeric::zeros(shape, device)
    }

    fn float_full<const D: usize>(
        shape: Shape<D>,
        fill_value: FloatElem<Self>,
        device: &R::Device,
    ) -> FloatTensor<Self, D> {
        numeric::full(shape, device, fill_value)
    }

    fn float_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        numeric::ones(shape, device)
    }

    fn float_sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::sub(lhs, rhs)
    }

    fn float_sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::sub_scalar(lhs, rhs)
    }

    fn float_mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::mul(lhs, rhs)
    }

    fn float_mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::mul_scalar(lhs, rhs)
    }

    fn float_div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::div(lhs, rhs)
    }

    fn float_div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::div_scalar(lhs, rhs)
    }

    fn float_remainder_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        numeric::remainder_scalar(lhs, rhs)
    }

    fn float_matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        matmul(lhs, rhs, MatmulStrategy::default())
    }

    fn float_swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        super::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        super::reshape(tensor, shape)
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::gather(dim, tensor, indices)
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::scatter(dim, tensor, indices, value)
    }

    fn float_select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        kernel::select(tensor, dim, indices)
    }

    fn float_select_assign<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::select_assign(tensor, dim, indices, value)
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        kernel::slice(tensor, ranges)
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        kernel::mask_where_auto(tensor, mask, value)
    }

    fn float_mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        kernel::mask_fill_auto(tensor, mask, value)
    }

    fn float_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::equal(lhs, rhs)
    }

    fn float_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::equal_elem(lhs, rhs)
    }

    fn float_greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater(lhs, rhs)
    }

    fn float_greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_elem(lhs, rhs)
    }

    fn float_greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal(lhs, rhs)
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::greater_equal_elem(lhs, rhs)
    }

    fn float_lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower(lhs, rhs)
    }

    fn float_lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_elem(lhs, rhs)
    }

    fn float_lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal(lhs, rhs)
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        kernel::lower_equal_elem(lhs, rhs)
    }

    fn float_sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        reduce::sum(tensor, Default::default())
    }

    fn float_sum_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        reduce::sum_dim(tensor, dim, Default::default())
    }

    fn float_mean_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        reduce::mean_dim(tensor, dim, Default::default())
    }

    fn float_prod<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        reduce::prod(tensor, Default::default())
    }

    fn float_prod_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        reduce::prod_dim(tensor, dim, Default::default())
    }

    fn float_exp<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, input| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::exp(input)
            }
            execute::__expand::<C>(context, input)
        })
    }

    fn float_log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::log(input)
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    fn float_log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::log1p(input)
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    fn float_powf_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: f32,
    ) -> FloatTensor<Self, D> {
        unary_op!(float(lhs, rhs.elem::<F>()) => |context, tensor, scalar| {
            #[cube]
            fn execute<C: Float>(input: C, scalar: C) -> C {
                C::powf(input, scalar)
            }
            execute::__expand::<C>(context, tensor, scalar)
        })
    }

    fn float_sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::sqrt(input)
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    fn float_abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::abs(input)
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    fn float_cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::cos(input)
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    fn float_sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::sin(input)
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    fn float_tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::tanh(input)
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    fn float_erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::erf(input)
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    fn float_argmax<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        reduce::argmax(tensor, dim, Default::default())
    }

    fn float_argmin<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        reduce::argmin(tensor, dim, Default::default())
    }

    fn float_into_int<const D: usize>(tensor: FloatTensor<Self, D>) -> IntTensor<Self, D> {
        kernel::cast(tensor)
    }

    fn float_clamp<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        kernel::clamp(tensor, min, max)
    }

    fn float_recip<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_op!(float(tensor) => |context, tensor| {
            #[cube]
            fn execute<C: Float>(input: C) -> C {
                C::recip(input)
            }
            execute::__expand::<C>(context, tensor)
        })
    }

    fn float_repeat_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> FloatTensor<Self, D> {
        kernel::repeat_dim(tensor, dim, times)
    }

    fn float_powf<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        numeric::pow(lhs, rhs)
    }

    fn float_permute<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: [usize; D],
    ) -> FloatTensor<Self, D> {
        permute(tensor, axes)
    }

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        expand(tensor, shape)
    }

    fn float_flip<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: &[usize],
    ) -> FloatTensor<Self, D> {
        kernel::flip(tensor, axes)
    }
}
