use super::{expand, numeric, permute};
use crate::kernel::prng::{random_bernoulli, random_normal, random_uniform};
use crate::kernel::unary_basic::BasicFloatUnaryKind;
use crate::kernel::{
    self, launch_unary_float, reduce, unary_basic, FloatUnaryOp, FloatUnaryOpFamily,
};
use crate::{
    element::BoolElement,
    kernel::matmul::{matmul, MatmulStrategy},
};
use crate::{execute_with_dtype, CubeBackend};
use crate::{CubeRuntime, FloatElement, IntElement};
use burn_tensor::ops::{BoolTensor, Device, FloatElem, FloatTensor, IntTensor};
use burn_tensor::{ops::FloatTensorOps, Distribution, Shape, TensorData};
use burn_tensor::{DType, ElementConversion, FloatDType};
use cubecl::prelude::*;
use half::{bf16, f16};
use std::ops::Range;

impl<R, F, I, BT> FloatTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        match data.dtype {
            DType::F64 | DType::F32 | DType::F16 | DType::BF16 => {
                super::from_data::<R>(data, device)
            }
            _ => unimplemented!("Unsupported dtype for `float_from_data`"),
        }
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        match distribution {
            Distribution::Default => random_uniform(shape, device, 0.elem::<F>(), 1.elem()),
            Distribution::Uniform(low, high) => {
                random_uniform(shape, device, low.elem::<F>(), high.elem())
            }
            Distribution::Bernoulli(prob) => random_bernoulli(shape, device, prob.elem::<F>()),
            Distribution::Normal(mean, std) => {
                random_normal(shape, device, mean.elem::<F>(), std.elem())
            }
        }
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            super::into_data::<R, E>(tensor).await
        )
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        super::to_device(tensor, device)
    }

    fn float_empty(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        super::empty::<R, F>(shape, device)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            numeric::add::<R, E>(lhs, rhs)
        )
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            numeric::add_scalar::<R, E>(lhs, rhs.elem())
        )
    }

    fn float_zeros(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        numeric::zeros::<R, F>(shape, device)
    }

    fn float_full(
        shape: Shape,
        fill_value: FloatElem<Self>,
        device: &R::Device,
    ) -> FloatTensor<Self> {
        numeric::full::<R, F>(shape, device, fill_value)
    }

    fn float_ones(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        numeric::ones::<R, F>(shape, device)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            numeric::sub::<R, E>(lhs, rhs)
        )
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            numeric::sub_scalar::<R, E>(lhs, rhs.elem())
        )
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            numeric::mul::<R, E>(lhs, rhs)
        )
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            numeric::mul_scalar::<R, E>(lhs, rhs.elem())
        )
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            numeric::div::<R, E>(lhs, rhs)
        )
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            numeric::div_scalar::<R, E>(lhs, rhs.elem())
        )
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            numeric::remainder::<R, E>(lhs, rhs)
        )
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            numeric::remainder_scalar::<R, E>(lhs, rhs.elem())
        )
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            matmul::<R, E>(lhs, rhs, None, MatmulStrategy::default()).unwrap()
        )
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
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            kernel::gather::<R, E, I>(dim, tensor, indices)
        )
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype, value.dtype),
            E,
            kernel::scatter::<R, E, I>(dim, tensor, indices, value)
        )
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            kernel::select::<R, E, I>(tensor, dim, indices)
        )
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype, value.dtype),
            E,
            kernel::select_assign::<R, E, I>(tensor, dim, indices, value)
        )
    }

    fn float_slice(tensor: FloatTensor<Self>, ranges: &[Range<usize>]) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            kernel::slice::<R, E>(tensor, ranges)
        )
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[Range<usize>],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype, value.dtype),
            E,
            kernel::slice_assign::<R, E>(tensor, ranges, value)
        )
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype, value.dtype),
            E,
            kernel::mask_where_auto::<R, E, BT>(tensor, mask, value)
        )
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            kernel::mask_fill_auto::<R, E, BT>(tensor, mask, value.elem())
        )
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            kernel::equal::<R, E, BT>(lhs, rhs)
        )
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            kernel::equal_elem::<R, E, BT>(lhs, rhs.elem())
        )
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            kernel::greater::<R, E, BT>(lhs, rhs)
        )
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            kernel::greater_elem::<R, E, BT>(lhs, rhs.elem())
        )
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            kernel::greater_equal::<R, E, BT>(lhs, rhs)
        )
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            kernel::greater_equal_elem::<R, E, BT>(lhs, rhs.elem())
        )
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            kernel::lower::<R, E, BT>(lhs, rhs)
        )
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            kernel::lower_elem::<R, E, BT>(lhs, rhs.elem())
        )
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype, rhs.dtype),
            E,
            kernel::lower_equal::<R, E, BT>(lhs, rhs)
        )
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        execute_with_dtype!(
            float(lhs.dtype),
            E,
            kernel::lower_equal_elem::<R, E, BT>(lhs, rhs.elem())
        )
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            reduce::sum::<R, E>(tensor, Default::default()).unwrap()
        )
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            reduce::reduce_dim::<R, E, E, reduce::Sum>(tensor, dim, Default::default()).unwrap()
        )
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            reduce::reduce_dim::<R, E, E, reduce::Mean>(tensor, dim, Default::default()).unwrap()
        )
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            reduce::reduce::<R, E, E, reduce::Prod>(tensor, Default::default()).unwrap()
        )
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            reduce::reduce_dim::<R, E, E, reduce::Prod>(tensor, dim, Default::default()).unwrap()
        )
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Exp)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Log)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Log1p)
    }

    fn float_powf_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        struct Powf;

        #[cube]
        impl<F: Float> FloatUnaryOp<F> for Powf {
            type Options = F;

            fn execute(input: Line<F>, options: &Self::Options) -> Line<F> {
                Line::powf(input, Line::new(*options))
            }
        }

        impl FloatUnaryOpFamily for Powf {
            type Options<F: Float> = F;
            type Unary<F: Float> = Self;
        }

        execute_with_dtype!(
            float(lhs.dtype),
            F,
            launch_unary_float::<R, F, Powf, _>(lhs, |_| ScalarArg::new(rhs.elem::<F>()))
        )
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Sqrt)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Abs)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Cos)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Sin)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Tanh)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Round)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Floor)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Ceil)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Erf)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            reduce::reduce_dim::<R, E, I, reduce::ArgMax>(tensor, dim, Default::default()).unwrap()
        )
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            reduce::reduce_dim::<R, E, I, reduce::ArgMin>(tensor, dim, Default::default()).unwrap()
        )
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        execute_with_dtype!(float(tensor.dtype), E, kernel::cast::<R, E, I>(tensor))
    }

    fn float_clamp(
        tensor: FloatTensor<Self>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            kernel::clamp::<R, E>(tensor, min.elem(), max.elem())
        )
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| &BasicFloatUnaryKind::Recip)
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            kernel::repeat_dim::<R, E>(tensor, dim, times)
        )
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_dtype!(float(lhs.dtype), E, numeric::pow::<R, E>(lhs, rhs))
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        permute(tensor, axes)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        expand(tensor, shape)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        execute_with_dtype!(
            float(tensor.dtype),
            E,
            kernel::flip::<R, E, BT>(tensor, axes)
        )
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        match (tensor.dtype, dtype) {
            (DType::F64, FloatDType::F64)
            | (DType::F32, FloatDType::F32)
            | (DType::BF16, FloatDType::BF16)
            | (DType::F16, FloatDType::F16) => tensor,
            (DType::F64, FloatDType::F32) => kernel::cast::<R, f64, f32>(tensor),
            (DType::F64, FloatDType::F16) => kernel::cast::<R, f64, f16>(tensor),
            (DType::F64, FloatDType::BF16) => kernel::cast::<R, f64, bf16>(tensor),
            (DType::F32, FloatDType::F64) => kernel::cast::<R, f32, f64>(tensor),
            (DType::F32, FloatDType::F16) => kernel::cast::<R, f32, f16>(tensor),
            (DType::F32, FloatDType::BF16) => kernel::cast::<R, f32, bf16>(tensor),
            (DType::F16, FloatDType::F64) => kernel::cast::<R, f16, f64>(tensor),
            (DType::F16, FloatDType::F32) => kernel::cast::<R, f16, f32>(tensor),
            (DType::F16, FloatDType::BF16) => kernel::cast::<R, f16, bf16>(tensor),
            (DType::BF16, FloatDType::F64) => kernel::cast::<R, bf16, f64>(tensor),
            (DType::BF16, FloatDType::F32) => kernel::cast::<R, bf16, f32>(tensor),
            (DType::BF16, FloatDType::F16) => kernel::cast::<R, bf16, f16>(tensor),
            _ => unimplemented!("Unsupported floating point type cast"),
        }
    }
}
