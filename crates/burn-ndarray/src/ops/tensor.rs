// Language
use alloc::vec::Vec;
use burn_backend::backend::ExecutionError;
use burn_backend::ops::GridSampleOptions;
use burn_backend::tensor::FloatTensor;
use burn_backend::{TensorMetadata, element::cast::ToElement};

// Current crate
use super::{
    NdArrayMathOps, NdArrayOps,
    matmul::{cross, matmul},
};
use crate::{
    NdArray, cast_to_dtype, cat_with_dtype, execute_with_int_dtype, tensor::NdArrayTensor,
};
use crate::{NdArrayDevice, SEED};
use crate::{
    SharedArray,
    element::{ExpElement, FloatNdArrayElement, IntNdArrayElement, QuantElement},
};
use crate::{execute_with_float_dtype, ops::grid_sample::grid_sample_2d};

// Workspace crates
use crate::rand::get_seeded_rng;
use burn_backend::{Distribution, FloatDType};
use burn_backend::{ElementConversion, Shape, TensorData, backend::Backend, ops::FloatTensorOps};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use libm::erf;

#[cfg(feature = "std")]
#[allow(dead_code)]
fn round_ties_even_wrapper(x: f64) -> f64 {
    x.round_ties_even()
}

#[cfg(not(feature = "std"))]
#[allow(dead_code)]
fn round_ties_even_wrapper(x: f64) -> f64 {
    if (x - x.floor()) == 0.5 {
        (x * 0.5).round() * 2.0
    } else {
        x.round()
    }
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> FloatTensorOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    fn float_from_data(data: TensorData, _device: &NdArrayDevice) -> FloatTensor<Self> {
        NdArrayTensor::from_data(data)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> FloatTensor<Self> {
        let mut seed = SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            get_seeded_rng()
        };
        let tensor = Self::float_from_data(
            TensorData::random::<E, _, _>(shape, distribution, &mut rng),
            device,
        );
        *seed = Some(rng);
        tensor
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn float_to_device(tensor: FloatTensor<Self>, _device: &NdArrayDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(
        shape: Shape,
        device: &<NdArray<E> as Backend>::Device,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        Self::float_zeros(shape, device, dtype)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::add)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::add_scalar(array, rhs.elem())
        })
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::sub)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::sub_scalar(array, rhs.elem())
        })
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::mul)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::mul_scalar(array, rhs.elem())
        })
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::div)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::div_scalar(array, rhs.elem())
        })
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::remainder)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::remainder_scalar(array, rhs.elem())
        })
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), matmul)
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| cross(lhs, rhs, dim))
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        Self::float_mul_scalar(tensor, (-1f32).elem::<E>())
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::recip(array)
        })
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayOps::swap_dims(array, dim1, dim2)
        })
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayOps::reshape(array, shape)
        })
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: NdArrayTensor,
    ) -> FloatTensor<Self> {
        execute_with_int_dtype!(
            indices,
            IntElem,
            |idx_array: SharedArray<IntElem>| -> NdArrayTensor {
                execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
                    NdArrayOps::gather(dim, array, idx_array)
                })
            }
        )
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: NdArrayTensor,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_int_dtype!(
            indices,
            IntElem,
            |idx_array: SharedArray<IntElem>| -> NdArrayTensor {
                execute_with_float_dtype!((tensor, value), |tensor, value| NdArrayOps::scatter(
                    dim, tensor, idx_array, value
                ))
            }
        )
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: NdArrayTensor,
    ) -> FloatTensor<Self> {
        execute_with_int_dtype!(
            indices,
            IntElem,
            |idx_array: SharedArray<IntElem>| -> NdArrayTensor {
                execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
                    NdArrayMathOps::select(array, dim, idx_array)
                })
            }
        )
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: NdArrayTensor,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_int_dtype!(
            indices,
            IntElem,
            |idx_array: SharedArray<IntElem>| -> NdArrayTensor {
                execute_with_float_dtype!((tensor, value), |tensor, value| {
                    NdArrayMathOps::select_assign(tensor, dim, idx_array, value)
                })
            }
        )
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[burn_backend::Slice]) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayOps::slice(array, slices)
        })
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[burn_backend::Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!((tensor, value), |tensor, value| {
            NdArrayOps::slice_assign(tensor, slices, value)
        })
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: NdArrayTensor,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!((tensor, value), |tensor, value| {
            NdArrayOps::mask_where(tensor, mask.bool(), value)
        })
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: NdArrayTensor,
        value: E,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayOps::mask_fill(array, mask.bool(), value.elem())
        })
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| { NdArrayMathOps::equal(lhs, rhs) })
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::equal_elem(array, rhs.elem())
        })
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| { NdArrayMathOps::greater(lhs, rhs) })
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::greater_elem(array, rhs.elem())
        })
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| {
            NdArrayMathOps::greater_equal(lhs, rhs)
        })
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::greater_equal_elem(array, rhs.elem())
        })
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| { NdArrayMathOps::lower(lhs, rhs) })
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::lower_elem(array, rhs.elem())
        })
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| {
            NdArrayMathOps::lower_equal(lhs, rhs)
        })
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::lower_equal_elem(array, rhs.elem())
        })
    }

    fn float_detach(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        tensor
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // Use view() for zero-copy on borrowed storage
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::mean_view(array.view())
        })
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // Use view() for zero-copy on borrowed storage
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::sum_view(array.view())
        })
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::mean_dim(array, dim)
        })
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::cumsum(array, dim)
        })
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::cumprod(array, dim)
        })
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::cummin(array, dim)
        })
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::cummax(array, dim)
        })
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::sum_dim(array, dim)
        })
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> NdArrayTensor {
        // Use view() for zero-copy on borrowed storage
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::argmax_view::<I>(array.view(), dim)
        })
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> NdArrayTensor {
        // Use view() for zero-copy on borrowed storage
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::argmin_view::<I>(array.view(), dim)
        })
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array.mapv_into(|a: FloatElem| a.exp_elem()).into_shared()
        })
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array.mapv_into(|a: FloatElem| a.log_elem()).into_shared()
        })
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // Use view() for zero-copy on borrowed storage
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::prod_view(array.view())
        })
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::prod_dim(array, dim)
        })
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // Use view() for zero-copy on borrowed storage
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::max_view(array.view())
        })
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        // Use view() for zero-copy on borrowed storage
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::min_view(array.view())
        })
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array.mapv_into(|a: FloatElem| a.log1p_elem()).into_shared()
        })
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array
                .mapv_into(|a: FloatElem| a.powf_elem(value))
                .into_shared()
        })
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array.mapv_into(|a: FloatElem| a.sqrt_elem()).into_shared()
        })
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::abs(array)
        })
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array
                .mapv_into(|a: FloatElem| (a.to_f64()).cos().elem())
                .into_shared()
        })
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array
                .mapv_into(|a: FloatElem| (a.to_f64()).sin().elem())
                .into_shared()
        })
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array
                .mapv_into(|a: FloatElem| (a.to_f64()).tanh().elem())
                .into_shared()
        })
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array
                .mapv_into(|a: FloatElem| round_ties_even_wrapper(a.to_f64()).elem())
                .into_shared()
        })
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array
                .mapv_into(|a: FloatElem| (a.to_f64()).floor().elem())
                .into_shared()
        })
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array
                .mapv_into(|a: FloatElem| (a.to_f64()).ceil().elem())
                .into_shared()
        })
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array
                .mapv_into(|a: FloatElem| (a.to_f64()).trunc().elem())
                .into_shared()
        })
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array
                .mapv_into(|a: FloatElem| erf(a.to_f64()).elem())
                .into_shared()
        })
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        cat_with_dtype!(tensors, dim, [F64, F32])
    }

    fn float_clamp_min(tensor: FloatTensor<Self>, min: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::clamp_min(array, min.elem())
        })
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::clamp_max(array, max.elem())
        })
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: E, max: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::clamp(array, min.elem(), max.elem())
        })
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            array.mapv(|a: FloatElem| a.elem::<I>()).into_shared()
        })
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), FloatElem, |lhs, rhs| {
            NdArrayMathOps::elementwise_op(lhs, rhs, |a: &FloatElem, b: &FloatElem| a.powf(*b))
        })
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayOps::permute(array, axes)
        })
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayOps::flip(array, axes)
        })
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayMathOps::sign_op(array)
        })
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayOps::expand(array, shape)
        })
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            cast_to_dtype(array, dtype.into())
        })
    }

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        options: GridSampleOptions,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!((tensor, grid), |tensor, grid| grid_sample_2d(
            tensor, grid, options
        ))
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, FloatElem, |array: SharedArray<FloatElem>| {
            NdArrayOps::unfold(array, dim, size, step)
        })
    }
}
