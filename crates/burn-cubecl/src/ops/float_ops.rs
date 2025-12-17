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
use burn_backend::tensor::{BoolTensor, Device, FloatElem, FloatTensor, IntTensor};
use burn_backend::{Backend, ExecutionError};
use burn_backend::{DType, ElementConversion, FloatDType, Slice};
use burn_backend::{Distribution, Shape, TensorData, ops::FloatTensorOps};
use cubecl::prelude::*;
use cubecl::std::scalar::InputScalar;
use cubek::reduce::components::instructions::ReduceOperationConfig;
use std::ops::Range;

impl<R, F, I, BT> FloatTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    #[tracing::instrument(
        skip(data),
        fields(?data.shape, ?data.dtype)
    )]
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        match data.dtype {
            DType::F64 | DType::F32 | DType::F16 | DType::BF16 => super::from_data(data, device),
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
            Distribution::Bernoulli(prob) => random_bernoulli(shape, device, prob as f32, dtype),
            Distribution::Normal(mean, std) => {
                random_normal(shape, device, mean.elem(), std.elem(), dtype)
            }
        }
    }

    #[tracing::instrument(
        skip(tensor),
        fields(from = ?tensor.device, shape = ?tensor.shape, dtype = ?tensor.dtype)
    )]
    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        super::into_data(tensor).await
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    #[tracing::instrument(
        skip(tensor),
        fields(from = ?tensor.device, shape = ?tensor.shape, dtype = ?tensor.dtype)
    )]
    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        super::to_device(tensor, device)
    }

    fn float_empty(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let dtype = dtype.into();
        super::empty(shape, device, dtype)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::add(lhs, rhs)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::add_scalar(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_zeros(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let dtype = dtype.into();
        numeric::zeros(device.clone(), shape, dtype)
    }

    fn float_full(
        shape: Shape,
        fill_value: FloatElem<Self>,
        device: &R::Device,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        let dtype: DType = dtype.into();
        let client = R::client(device);
        numeric::full_device_dtype(
            client,
            shape,
            device.clone(),
            InputScalar::new(fill_value, dtype),
            dtype,
        )
    }

    fn float_ones(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let dtype = dtype.into();
        numeric::ones(device.clone(), shape, dtype)
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::sub(lhs, rhs)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::sub_scalar(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::mul(lhs, rhs)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::mul_scalar(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::div(lhs, rhs)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::div_scalar(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        numeric::remainder(lhs, rhs)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        numeric::remainder_scalar(lhs, InputScalar::new(rhs, dtype))
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let dtype = lhs.dtype;
        matmul(lhs, rhs, None, MatmulStrategy::default(), dtype).unwrap()
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        kernel::cross(lhs, rhs, dim)
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

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::scatter(dim, tensor, indices, value, false)
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::select(tensor, dim, indices)
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::select_assign(tensor, dim, indices, value, false)
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        // Check if all steps are 1
        let all_steps_one = slices.iter().all(|info| info.step == 1);

        if all_steps_one {
            // Use optimized slice for step=1
            let simple_ranges: Vec<Range<usize>> = slices
                .iter()
                .enumerate()
                .map(|(i, slice)| slice.to_range(tensor.shape[i]))
                .collect();

            kernel::slice(tensor, &simple_ranges)
        } else {
            // Use slice with steps kernel
            kernel::slice_with_steps(tensor, slices)
        }
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        kernel::mask_where_auto(tensor, mask, value, BT::dtype())
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        let dtype = tensor.dtype;
        kernel::mask_fill_auto(tensor, mask, InputScalar::new(value, dtype), BT::dtype())
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::equal(lhs, rhs, BT::dtype())
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::equal_elem(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::greater(lhs, rhs, BT::dtype())
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::greater_elem(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::greater_equal(lhs, rhs, BT::dtype())
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::greater_equal_elem(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::lower(lhs, rhs, BT::dtype())
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::lower_elem(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::lower_equal(lhs, rhs, BT::dtype())
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::lower_equal_elem(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let tensor = into_contiguous(tensor);
        reduce::sum_fallback(tensor, Default::default()).unwrap()
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::reduce(tensor, None, Default::default(), ReduceOperationConfig::Max).unwrap()
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim(
            tensor,
            None,
            dim,
            Default::default(),
            ReduceOperationConfig::Max,
        )
        .unwrap()
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::reduce(tensor, None, Default::default(), ReduceOperationConfig::Min).unwrap()
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim(
            tensor,
            None,
            dim,
            Default::default(),
            ReduceOperationConfig::Min,
        )
        .unwrap()
    }

    fn float_max_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::reduce(
            tensor,
            None,
            Default::default(),
            ReduceOperationConfig::MaxAbs,
        )
        .unwrap()
    }

    fn float_max_abs_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim(
            tensor,
            None,
            dim,
            Default::default(),
            ReduceOperationConfig::MaxAbs,
        )
        .unwrap()
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim(
            tensor,
            None,
            dim,
            Default::default(),
            ReduceOperationConfig::Sum,
        )
        .unwrap()
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim(
            tensor,
            None,
            dim,
            Default::default(),
            ReduceOperationConfig::Mean,
        )
        .unwrap()
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        numeric::cumsum(tensor, dim)
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        numeric::cumprod(tensor, dim)
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        numeric::cummin(tensor, dim)
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        numeric::cummax(tensor, dim)
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        reduce::reduce(
            tensor,
            None,
            Default::default(),
            ReduceOperationConfig::Prod,
        )
        .unwrap()
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce::reduce_dim(
            tensor,
            None,
            dim,
            Default::default(),
            ReduceOperationConfig::Prod,
        )
        .unwrap()
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
        reduce::reduce_dim(
            tensor,
            Some(<Self as Backend>::IntElem::dtype()),
            dim,
            Default::default(),
            ReduceOperationConfig::ArgMax,
        )
        .unwrap()
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim(
            tensor,
            Some(<Self as Backend>::IntElem::dtype()),
            dim,
            Default::default(),
            ReduceOperationConfig::ArgMin,
        )
        .unwrap()
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        kernel::cast(tensor, I::dtype())
    }

    fn float_clamp(
        tensor: FloatTensor<Self>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        let dtype = tensor.dtype;
        kernel::clamp(
            tensor,
            InputScalar::new(min, dtype),
            InputScalar::new(max, dtype),
        )
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_basic::launch::<R, _>(tensor, |_| BasicFloatUnaryKind::Recip)
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
        kernel::flip(tensor, axes, BT::dtype())
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        kernel::cast(tensor, dtype.into())
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
        kernel::is_nan(tensor, BT::dtype())
    }

    fn float_is_inf(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        kernel::is_inf(tensor, BT::dtype())
    }
}
