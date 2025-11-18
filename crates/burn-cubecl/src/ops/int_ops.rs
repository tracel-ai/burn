use self::unary_basic_int::BasicIntUnaryKind;

use super::{expand, numeric, permute, unfold};
use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    kernel::{
        self,
        matmul::{MatmulStrategy, matmul},
    },
};
use crate::{
    element::BoolElement,
    kernel::prng::{random_bernoulli, random_normal, random_uniform},
};
use crate::{
    execute_with_dtype,
    kernel::{
        BitwiseShlOp, BitwiseShrOp, NumericUnaryOp, NumericUnaryOpFamily, launch_binop_int,
        launch_scalar_binop_int, launch_unary_numeric, reduce, unary_basic_int,
    },
};
use burn_tensor::ops::{BoolTensor, Device, FloatTensor, IntElem, IntTensor};
use burn_tensor::{DType, IntDType};
use burn_tensor::{Distribution, ElementConversion, Shape, TensorData, ops::IntTensorOps};
use cubecl::prelude::*;
use cubecl::reduce::instructions::ReduceFnConfig;
use cubecl::{frontend::Numeric, std::scalar::InputScalar};
use std::ops::Range;

impl<R, F, I, BT> IntTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn int_empty(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let dtype = dtype.into();
        execute_with_dtype!(int(dtype), I, super::empty::<R, I>(shape, device))
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> TensorData {
        execute_with_dtype!(int(tensor.dtype), I, super::into_data::<R, I>(tensor).await)
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        match data.dtype {
            DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8 => super::from_data::<R>(data, device),
            _ => unimplemented!("Unsupported dtype for `int_from_data`"),
        }
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

    fn int_slice(tensor: IntTensor<Self>, slices: &[burn_tensor::Slice]) -> IntTensor<Self> {
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

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        ranges: &[burn_tensor::Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::slice_assign::<R>(tensor, ranges, value)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        matmul::<R>(lhs, rhs, None, MatmulStrategy::default(), dtype).unwrap()
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::mask_where_auto::<R>(tensor, mask, value, BT::dtype())
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntElem<Self>,
    ) -> IntTensor<Self> {
        let dtype = tensor.dtype;
        kernel::mask_fill_auto::<R>(tensor, mask, InputScalar::new(value, dtype), BT::dtype())
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::gather::<R>(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::scatter::<R>(dim, tensor, indices, value)
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::select::<R>(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        kernel::select_assign::<R>(tensor, dim, indices, value, false)
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::equal::<R>(lhs, rhs, BT::dtype())
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::equal_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::greater::<R>(lhs, rhs, BT::dtype())
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::greater_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::greater_equal::<R>(lhs, rhs, BT::dtype())
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::greater_equal_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::lower::<R>(lhs, rhs, BT::dtype())
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::lower_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        kernel::lower_equal::<R>(lhs, rhs, BT::dtype())
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let dtype = lhs.dtype;
        kernel::lower_equal_elem::<R>(lhs, InputScalar::new(rhs, dtype), BT::dtype())
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::add::<R>(lhs, rhs)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        numeric::add_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::sub::<R>(lhs, rhs)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        numeric::sub_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::mul::<R>(lhs, rhs)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        numeric::mul_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::div::<R>(lhs, rhs)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        numeric::div_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::remainder::<R>(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        numeric::remainder_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn int_zeros(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let dtype = dtype.into();
        execute_with_dtype!(int(dtype), I, numeric::zeros::<R, I>(shape, device))
    }

    fn int_ones(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let dtype = dtype.into();
        execute_with_dtype!(int(dtype), I, numeric::ones::<R, I>(shape, device))
    }

    fn int_full(
        shape: Shape,
        fill_value: IntElem<Self>,
        device: &Device<Self>,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        let dtype = dtype.into();
        execute_with_dtype!(
            int(dtype),
            I,
            numeric::full::<R, I>(shape, device, fill_value.elem())
        )
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce::sum_fallback::<R>(tensor, Default::default()).unwrap()
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Sum).unwrap()
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce::reduce::<R>(tensor, Default::default(), ReduceFnConfig::Prod).unwrap()
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Prod).unwrap()
    }

    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce::reduce::<R>(tensor, Default::default(), ReduceFnConfig::Max).unwrap()
    }

    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Max).unwrap()
    }

    fn int_max_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce::reduce::<R>(tensor, Default::default(), ReduceFnConfig::MaxAbs).unwrap()
    }

    fn int_max_abs_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::MaxAbs).unwrap()
    }

    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        reduce::reduce::<R>(tensor, Default::default(), ReduceFnConfig::Min).unwrap()
    }

    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Min).unwrap()
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::Mean).unwrap()
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_dtype!(int(tensor.dtype), I, numeric::cumsum::<R, I>(tensor, dim))
    }

    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_dtype!(int(tensor.dtype), I, numeric::cumprod::<R, I>(tensor, dim))
    }

    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_dtype!(int(tensor.dtype), I, numeric::cummin::<R, I>(tensor, dim))
    }

    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_dtype!(int(tensor.dtype), I, numeric::cummax::<R, I>(tensor, dim))
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::ArgMax).unwrap()
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce::reduce_dim::<R>(tensor, dim, Default::default(), ReduceFnConfig::ArgMin).unwrap()
    }

    fn int_clamp(
        tensor: IntTensor<Self>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self> {
        execute_with_dtype!(
            int(tensor.dtype),
            I,
            kernel::clamp::<R, I>(tensor, min.elem(), max.elem())
        )
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
            type Options = ();
            type Unary<N: Numeric> = Self;
        }

        launch_unary_numeric::<R, Abs, _>(tensor, |_| ())
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        kernel::cast::<R>(tensor, F::dtype())
    }

    fn int_swap_dims(mut tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        tensor.strides.swap(dim1, dim2);
        tensor.shape = tensor.shape.swap(dim1, dim2).unwrap();

        tensor
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        execute_with_dtype!(
            int(tensor.dtype),
            I,
            kernel::repeat_dim::<R, I>(tensor, dim, times)
        )
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
        let dtype = IntElem::<Self>::dtype();
        match distribution {
            Distribution::Default => random_uniform(shape, device, 0., 255., dtype),
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

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        permute(tensor, axes)
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        expand(tensor, shape)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        kernel::flip::<R>(tensor, axes, BT::dtype())
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::bitwise_and::<R>(lhs, rhs)
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        numeric::bitwise_and_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::bitwise_or::<R>(lhs, rhs)
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        numeric::bitwise_or_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        numeric::bitwise_xor::<R>(lhs, rhs)
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        numeric::bitwise_xor_scalar::<R>(lhs, InputScalar::new(rhs, dtype))
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unary_basic_int::launch::<R, _>(tensor, |_| BasicIntUnaryKind::BitwiseNot)
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        launch_binop_int::<R, kernel::BitwiseShlOp>(lhs, rhs)
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        launch_scalar_binop_int::<R, BitwiseShlOp>(lhs, InputScalar::new(rhs, dtype))
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        launch_binop_int::<R, BitwiseShrOp>(lhs, rhs)
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let dtype = lhs.dtype;
        launch_scalar_binop_int::<R, BitwiseShrOp>(lhs, InputScalar::new(rhs, dtype))
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        kernel::cast::<R>(tensor, dtype.into())
    }

    fn int_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        unfold(tensor, dim, size, step)
    }
}
