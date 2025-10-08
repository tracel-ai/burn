// Language
use alloc::vec::Vec;
use burn_tensor::ops::FloatTensor;
use burn_tensor::ops::InterpolateMode;
use burn_tensor::{TensorMetadata, cast::ToElement};

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
use burn_tensor::{Distribution, FloatDType};
use burn_tensor::{ElementConversion, Shape, TensorData, backend::Backend, ops::FloatTensorOps};

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

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        tensor.into_data()
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
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::add_scalar(lhs, rhs.elem()))
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::sub)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::sub_scalar(lhs, rhs.elem()))
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::mul)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::mul_scalar(lhs, rhs.elem()))
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::div)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::div_scalar(lhs, rhs.elem()))
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::remainder)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::remainder_scalar(lhs, rhs.elem()))
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
        execute_with_float_dtype!(tensor, NdArrayMathOps::recip)
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::swap_dims(tensor, dim1, dim2))
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::reshape(tensor, shape))
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: NdArrayTensor,
    ) -> FloatTensor<Self> {
        execute_with_int_dtype!(indices, I, |indices| -> NdArrayTensor {
            execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::gather(
                dim, tensor, indices
            ))
        })
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: NdArrayTensor,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_int_dtype!(indices, I, |indices| -> NdArrayTensor {
            execute_with_float_dtype!((tensor, value), |tensor, value| NdArrayMathOps::scatter(
                dim, tensor, indices, value
            ))
        })
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: NdArrayTensor,
    ) -> FloatTensor<Self> {
        execute_with_int_dtype!(indices, I, |indices| -> NdArrayTensor {
            execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::select(
                tensor, dim, indices
            ))
        })
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: NdArrayTensor,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_int_dtype!(indices, I, |indices| -> NdArrayTensor {
            execute_with_float_dtype!((tensor, value), |tensor, value| {
                NdArrayMathOps::select_assign(tensor, dim, indices, value)
            })
        })
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[burn_tensor::Slice]) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::slice(tensor, slices))
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[burn_tensor::Slice],
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
            NdArrayMathOps::mask_where(tensor, mask.bool(), value)
        })
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: NdArrayTensor,
        value: E,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::mask_fill(
            tensor,
            mask.bool(),
            value.elem()
        ))
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| { NdArrayMathOps::equal(lhs, rhs) })
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, |tensor| {
            NdArrayMathOps::equal_elem(tensor, rhs.elem())
        })
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| { NdArrayMathOps::greater(lhs, rhs) })
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, |tensor| {
            NdArrayMathOps::greater_elem(tensor, rhs.elem())
        })
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| {
            NdArrayMathOps::greater_equal(lhs, rhs)
        })
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, |tensor| {
            NdArrayMathOps::greater_equal_elem(tensor, rhs.elem())
        })
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| { NdArrayMathOps::lower(lhs, rhs) })
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, |tensor| {
            NdArrayMathOps::lower_elem(tensor, rhs.elem())
        })
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!((lhs, rhs), |lhs, rhs| {
            NdArrayMathOps::lower_equal(lhs, rhs)
        })
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor {
        execute_with_float_dtype!(lhs, |tensor| {
            NdArrayMathOps::lower_equal_elem(tensor, rhs.elem())
        })
    }

    fn float_detach(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        tensor
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, NdArrayMathOps::mean)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, NdArrayMathOps::sum)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::mean_dim(tensor, dim))
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::cumsum(tensor, dim))
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::sum_dim(tensor, dim))
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> NdArrayTensor {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::argmax::<I>(tensor, dim))
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> NdArrayTensor {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::argmin::<I>(tensor, dim))
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor.mapv_into(|a| a.exp_elem()).into_shared()
        })
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor.mapv_into(|a| a.log_elem()).into_shared()
        })
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, NdArrayMathOps::prod)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::prod_dim(tensor, dim))
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor.mapv_into(|a| a.log1p_elem()).into_shared()
        })
    }

    fn float_powf_scalar(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            if value == 2.0 {
                // Happens often and is faster.
                tensor.mapv_into(|a| a * a).into_shared()
            } else if value.floor() == value {
                // Is faster then powf
                tensor
                    .mapv_into(|a| a.powi_elem(value as i32))
                    .into_shared()
            } else {
                // Default
                tensor.mapv_into(|a| a.powf_elem(value)).into_shared()
            }
        })
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor.mapv_into(|a| a.sqrt_elem()).into_shared()
        })
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, NdArrayMathOps::abs)
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor
                .mapv_into(|a| (a.to_f64()).cos().elem())
                .into_shared()
        })
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor
                .mapv_into(|a| (a.to_f64()).sin().elem())
                .into_shared()
        })
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor
                .mapv_into(|a| (a.to_f64()).tanh().elem())
                .into_shared()
        })
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor
                .mapv_into(|a| round_ties_even_wrapper(a.to_f64()).elem())
                .into_shared()
        })
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor
                .mapv_into(|a| (a.to_f64()).floor().elem())
                .into_shared()
        })
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor
                .mapv_into(|a| (a.to_f64()).ceil().elem())
                .into_shared()
        })
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: SharedArray<E>| {
            tensor.mapv_into(|a| erf(a.to_f64()).elem()).into_shared()
        })
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        cat_with_dtype!(tensors, dim, [F64, F32])
    }

    fn float_clamp_min(tensor: FloatTensor<Self>, min: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::clamp_min(
            tensor,
            min.elem()
        ))
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::clamp_max(
            tensor,
            max.elem()
        ))
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: E, max: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::clamp(
            tensor,
            min.elem(),
            max.elem()
        ))
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_float_dtype!(tensor, |tensor: SharedArray<E>| {
            tensor.mapv(|a| a.elem::<I>()).into_shared()
        })
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), E, |lhs, rhs| NdArrayMathOps::elementwise_op(
            lhs,
            rhs,
            |a: &E, b: &E| a.powf(*b)
        ))
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::permute(tensor, axes))
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::flip(tensor, axes))
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, NdArrayMathOps::sign_op)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::expand(tensor, shape))
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| cast_to_dtype(tensor, dtype.into()))
    }

    fn float_grid_sample_2d(
        tensor: FloatTensor<Self>,
        grid: FloatTensor<Self>,
        method: InterpolateMode,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!((tensor, grid), |tensor, grid| grid_sample_2d(
            tensor, grid, method
        ))
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::unfold(tensor, dim, size, step))
    }
}
