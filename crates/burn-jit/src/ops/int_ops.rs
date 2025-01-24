use self::unary_basic_int::BasicIntUnaryKind;

use super::{expand, numeric, permute};
use crate::kernel::{
    launch_binop_int, launch_scalar_binop_int, launch_unary_numeric, reduce, unary_basic_int,
    BitwiseShlOp, BitwiseShrOp, NumericUnaryOp, NumericUnaryOpFamily,
};
use crate::{
    element::BoolElement,
    kernel::prng::{random_bernoulli, random_normal, random_uniform},
};
use crate::{kernel, FloatElement, IntElement, JitBackend, JitRuntime};
use burn_tensor::ops::{BoolTensor, Device, FloatTensor, IntElem, IntTensor};
use burn_tensor::{ops::IntTensorOps, Distribution, ElementConversion, Shape, TensorData};
use cubecl::frontend::Numeric;
use cubecl::prelude::*;
use std::ops::Range;

impl<R, F, I, BT> IntTensorOps<Self> for JitBackend<R, F, I, BT>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn int_empty(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        super::empty::<R, I>(shape, device)
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> TensorData {
        super::into_data::<R, I>(tensor).await
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        super::from_data::<R, I>(data, device)
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
        kernel::slice::<R, I>(tensor, ranges)
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        ranges: &[Range<usize>],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::slice_assign::<R, I>(tensor, ranges, value)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::mask_where_auto::<R, I, BT>(tensor, mask, value)
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntElem<Self>,
    ) -> IntTensor<Self> {
        kernel::mask_fill_auto::<R, I, BT>(tensor, mask, value)
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::gather::<R, I, I>(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::scatter::<R, I, I>(dim, tensor, indices, value)
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::select::<R, I, I>(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::select_assign::<R, I, I>(tensor, dim, indices, value)
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::equal::<R, I, BT>(lhs, rhs)
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::equal_elem::<R, I, BT>(lhs, rhs)
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::greater::<R, I, BT>(lhs, rhs)
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::greater_elem::<R, I, BT>(lhs, rhs)
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::greater_equal::<R, I, BT>(lhs, rhs)
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::greater_equal_elem::<R, I, BT>(lhs, rhs)
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::lower::<R, I, BT>(lhs, rhs)
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::lower_elem::<R, I, BT>(lhs, rhs)
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::lower_equal::<R, I, BT>(lhs, rhs)
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        kernel::lower_equal_elem::<R, I, BT>(lhs, rhs)
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::add::<R, I>(lhs, rhs)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::add_scalar::<R, I>(lhs, rhs)
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::sub::<R, I>(lhs, rhs)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::sub_scalar::<R, I>(lhs, rhs)
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::mul::<R, I>(lhs, rhs)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::mul_scalar::<R, I>(lhs, rhs)
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::div::<R, I>(lhs, rhs)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::div_scalar::<R, I>(lhs, rhs)
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::remainder::<R, I>(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::remainder_scalar::<R, I>(lhs, rhs)
    }

    fn int_zeros(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        numeric::zeros::<R, I>(shape, device)
    }

    fn int_ones(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        numeric::ones::<R, I>(shape, device)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce::sum::<R, I>(tensor, Default::default()).unwrap()
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R, I, I, reduce::Sum>(tensor, dim, Default::default()).unwrap()
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce::reduce::<R, I, I, reduce::Prod>(tensor, Default::default()).unwrap()
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R, I, I, reduce::Prod>(tensor, dim, Default::default()).unwrap()
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R, I, I, reduce::Mean>(tensor, dim, Default::default()).unwrap()
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R, I, I, reduce::ArgMax>(tensor, dim, Default::default()).unwrap()
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R, I, I, reduce::ArgMin>(tensor, dim, Default::default()).unwrap()
    }

    fn int_clamp(
        tensor: IntTensor<Self>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self> {
        kernel::clamp::<R, I>(tensor, min, max)
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        struct Abs;

        #[cube]
        impl<N: Numeric> NumericUnaryOp<N> for Abs {
            type Options = ();

            fn execute(input: Line<N>, _options: &Self::Options) -> Line<N> {
                Line::abs(input)
            }
        }

        impl NumericUnaryOpFamily for Abs {
            type Options<N: Numeric> = ();
            type Unary<N: Numeric> = Self;
        }

        launch_unary_numeric::<R, I, Abs, _>(tensor, |_| ())
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        kernel::cast::<R, I, F>(tensor)
    }

    fn int_swap_dims(mut tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape.dims.swap(dim1, dim2);

        tensor
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        kernel::repeat_dim::<R, I>(tensor, dim, times)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
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

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        permute(tensor, axes)
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        expand(tensor, shape)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        kernel::flip::<R, I, BT>(tensor, axes)
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::bitwise_and::<R, I>(lhs, rhs)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::bitwise_and_scalar::<R, I>(lhs, rhs)
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::bitwise_or::<R, I>(lhs, rhs)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::bitwise_or_scalar(lhs, rhs)
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::bitwise_xor::<R, I>(lhs, rhs)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        numeric::bitwise_xor_scalar(lhs, rhs)
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_basic_int::launch::<R, _, I>(tensor, |_| &BasicIntUnaryKind::BitwiseNot)
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let lhs_cast = kernel::cast::<R, I, u32>(lhs);
        let rhs_cast = kernel::cast::<R, I, u32>(rhs);
        launch_binop_int::<R, u32, kernel::BitwiseShlOp>(lhs_cast, rhs_cast)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let lhs_cast = kernel::cast::<R, I, u32>(lhs);
        let rhs_cast = rhs.elem::<u32>();
        launch_scalar_binop_int::<R, u32, BitwiseShlOp>(lhs_cast, rhs_cast)
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let lhs_cast = kernel::cast::<R, I, u32>(lhs);
        let rhs_cast = kernel::cast::<R, I, u32>(rhs);
        launch_binop_int::<R, u32, BitwiseShrOp>(lhs_cast, rhs_cast)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let lhs_cast = kernel::cast::<R, I, u32>(lhs);
        let rhs_cast = rhs.elem::<u32>();
        launch_scalar_binop_int::<R, u32, BitwiseShrOp>(lhs_cast, rhs_cast)
    }
}
