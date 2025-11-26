use super::{expand, numeric, permute, unfold};
use crate::CubeBackend;
use crate::kernel::prng::{random_bernoulli, random_normal, random_uniform};
use crate::kernel::{
    self, FloatUnaryOp, FloatUnaryOpFamily, launch_unary_float, reduce, unary_basic,
};
use crate::kernel::{into_contiguous, unary_basic::BasicFloatUnaryKind};
use crate::{CubeRuntime, FloatElement, IntElement};
use crate::{
    element::BoolElement,
    kernel::matmul::{MatmulStrategy, matmul},
};
use burn_tensor::ops::{BoolTensor, Device, FloatElem, FloatTensor, IntTensor};
use burn_tensor::{DType, ElementConversion, FloatDType};
use burn_tensor::{Distribution, Shape, TensorData, ops::FloatTensorOps};
use cubecl::prelude::*;
use cubecl::reduce::instructions::ReduceFnConfig;
use cubecl::std::scalar::InputScalar;
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
        let dtype = FloatElem::<Self>::dtype();
        match distribution {
            Distribution::Default => random_uniform(shape, device, 0., 1., dtype),
            Distribution::Uniform(low, high) => {
                random_uniform(shape, device, low.elem(), high.elem(), dtype)
            }
            Distribution::Bernoulli(prob) => {
                random_bernoulli::<R>(shape, device, prob as f32, dtype)
            }
            Distribution::Normal(mean, std) => {
                random_normal(shape, device, mean.elem(), std.elem(), dtype)
            }
        }
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        super::into_data::<R>(tensor).await
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        super::to_device(tensor, device)
    }

    fn float_empty(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let dtype = dtype.into();
        super::empty::<R>(shape, device, dtype)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::add::<R>(lhs, rhs)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::add_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_zeros(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let dtype = dtype.into();
        numeric::zeros::<R>(device.clone(), shape, dtype)
    }

    fn float_full(
        shape: Shape,
        fill_value: FloatElem<Self>,
        device: &R::Device,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        let dtype: DType = dtype.into();
        let client = R::client(device);
        numeric::full_device_dtype::<R>(
            client,
            shape,
            device.clone(),
            InputScalar::new(fill_value, dtype),
            dtype,
        )
    }

    fn float_ones(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let dtype = dtype.into();
        numeric::ones::<R>(device.clone(), shape, dtype)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::sub::<R>(lhs, rhs)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::sub_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::mul::<R>(lhs, rhs)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::mul_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::div::<R>(lhs, rhs)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::div_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::remainder::<R>(lhs, rhs)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::remainder_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        matmul::<R>(lhs, rhs, None, MatmulStrategy::default(), dtype).unwrap()
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        kernel::cross::<R>(lhs, rhs, dim)
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
        kernel::gather::<R>(dim, tensor, indices)
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::scatter::<R>(dim, tensor, indices, value)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::select::<R>(tensor, dim, indices)
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::select_assign::<R>(tensor, dim, indices, value, false)
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[burn_tensor::Slice]) -> FloatTensor<Self> {
        // Check if all steps are 1
        let all_steps_one = slices.iter().all(|info| info.step == 1);

        if all_steps_one {
            // Use optimized slice for step=1
            let simple_ranges: Vec<Range<usize>> = slices
                .iter()
                .enumerate()
                .map(|(i, slice)| slice.to_range(tensor.shape[i]))
                .collect();

            kernel::slice::<R>(tensor, &simple_ranges)
        } else {
            // Use slice with steps kernel
            kernel::slice_with_steps::<R>(tensor, slices)
        }
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[burn_tensor::Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::slice_assign::<R>(tensor, ranges, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::mask_where_auto::<R>(tensor, mask, value, BT::dtype())
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        let dtype = tensor.dtype;
        kernel::mask_fill_auto::<R>(tensor, mask, InputScalar::new(value, dtype), BT::dtype())
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::equal::<R>(lhs, rhs, BT::dtype())
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::equal_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::greater::<R>(lhs, rhs, BT::dtype())
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::greater_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::greater_equal::<R>(lhs, rhs, BT::dtype())
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::greater_equal_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::lower::<R>(lhs, rhs, BT::dtype())
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::lower_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::lower_equal::<R>(lhs, rhs, BT::dtype())
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::lower_equal_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let tensor = into_contiguous::<R>(tensor);
        reduce::sum_fallback::<R>(tensor, Default::default()).unwrap()
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::reduce::<R>(tensor, Default::default(), ReduceFnConfig::Max).unwrap()
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Max).unwrap()
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::reduce::<R>(tensor, Default::default(), ReduceFnConfig::Min).unwrap()
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Min).unwrap()
    }

    fn float_max_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::reduce::<R>(tensor, Default::default(), ReduceFnConfig::MaxAbs).unwrap()
    }

    fn float_max_abs_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::MaxAbs).unwrap()
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Sum).unwrap()
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Mean).unwrap()
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        numeric::cumsum::<R>(tensor, dim)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        numeric::cumprod::<R>(tensor, dim)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        numeric::cummin::<R>(tensor, dim)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        numeric::cummax::<R>(tensor, dim)
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::reduce::<R>(tensor, Default::default(), ReduceFnConfig::Prod).unwrap()
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Prod).unwrap()
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Exp)
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Log)
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Log1p)
    }

    fn float_powf_scalar_impl(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        struct Powf;

        #[cube]
        impl<F: Float> FloatUnaryOp<F> for Powf {
            type Options = InputScalar;

            fn execute(input: Line<F>, options: &Self::Options) -> Line<F> {
                Line::powf(input, Line::new(options.get::<F>()))
            }
        }

        impl FloatUnaryOpFamily for Powf {
            type Options = InputScalar;
            type Unary<F: Float> = Self;
        }

        let dtype = lhs.dtype;
        launch_unary_float::<R, Powf, _>(lhs, |_| InputScalar::new(rhs, dtype))
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Sqrt)
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Abs)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Cos)
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Sin)
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Tanh)
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Round)
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Floor)
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Ceil)
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Trunc)
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Erf)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::ArgMax).unwrap()
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::ArgMin).unwrap()
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        kernel::cast::<R>(tensor, I::dtype())
    }

    fn float_clamp(
        tensor: FloatTensor<Self>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        let dtype = tensor.dtype;
        kernel::clamp::<R>(
            tensor,
            InputScalar::new(min, dtype),
            InputScalar::new(max, dtype),
        )
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Recip)
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        kernel::repeat_dim::<R>(tensor, dim, times)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::pow::<R>(lhs, rhs)
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        permute(tensor, axes)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        expand(tensor, shape)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        kernel::flip::<R>(tensor, axes, BT::dtype())
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        kernel::cast::<R>(tensor, dtype.into())
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        unfold(tensor, dim, size, step)
    }

    fn float_is_nan(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::is_nan::<R>(tensor, BT::dtype())
    }

    fn float_is_inf(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::is_inf::<R>(tensor, BT::dtype())
    }
}
