// Language
use alloc::vec::Vec;
use core::ops::Range;
use ndarray::Zip;

// Current crate
use super::{matmul::matmul, NdArrayMathOps, NdArrayOps};
use crate::element::{FloatNdArrayElement, QuantElement};
use crate::{tensor::NdArrayTensor, NdArray};
use crate::{NdArrayDevice, SEED};

// Workspace crates
use burn_common::rand::get_seeded_rng;
use burn_tensor::Distribution;
use burn_tensor::{backend::Backend, ops::FloatTensorOps, ElementConversion, Shape, TensorData};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use libm::erf;

impl<E: FloatNdArrayElement, Q: QuantElement> FloatTensorOps<Self> for NdArray<E, Q> {
    fn float_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor<E> {
        NdArrayTensor::from_data(data)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> NdArrayTensor<E> {
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

    fn float_shape(tensor: &NdArrayTensor<E>) -> Shape {
        tensor.shape()
    }

    async fn float_into_data(tensor: NdArrayTensor<E>) -> TensorData {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        TensorData::new(values, shape)
    }

    fn float_device(_tensor: &NdArrayTensor<E>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn float_to_device(tensor: NdArrayTensor<E>, _device: &NdArrayDevice) -> NdArrayTensor<E> {
        tensor
    }

    fn float_empty(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<E> {
        NdArray::<E>::float_zeros(shape, device)
    }

    fn float_add(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        NdArrayMathOps::add(lhs, rhs)
    }

    fn float_add_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        NdArrayMathOps::add_scalar(lhs, rhs)
    }

    fn float_sub(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        NdArrayMathOps::sub(lhs, rhs)
    }

    fn float_sub_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        NdArrayMathOps::sub_scalar(lhs, rhs)
    }

    fn float_mul(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        NdArrayMathOps::mul(lhs, rhs)
    }

    fn float_mul_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        NdArrayMathOps::mul_scalar(lhs, rhs)
    }

    fn float_div(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        NdArrayMathOps::div(lhs, rhs)
    }

    fn float_div_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        NdArrayMathOps::div_scalar(lhs, rhs)
    }

    fn float_remainder_scalar(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<E> {
        NdArrayMathOps::remainder_scalar(lhs, rhs)
    }

    fn float_matmul(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        matmul(lhs, rhs)
    }

    fn float_neg(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        Self::float_mul_scalar(tensor, (-1f32).elem::<E>())
    }

    fn float_recip(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        NdArrayMathOps::recip(tensor)
    }

    fn float_swap_dims(tensor: NdArrayTensor<E>, dim1: usize, dim2: usize) -> NdArrayTensor<E> {
        NdArrayOps::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape(tensor: NdArrayTensor<E>, shape: Shape) -> NdArrayTensor<E> {
        NdArrayOps::reshape(tensor, shape)
    }

    fn float_gather(
        dim: usize,
        tensor: NdArrayTensor<E>,
        indices: NdArrayTensor<i64>,
    ) -> NdArrayTensor<E> {
        NdArrayMathOps::gather(dim, tensor, indices)
    }

    fn float_scatter(
        dim: usize,
        tensor: NdArrayTensor<E>,
        indices: NdArrayTensor<i64>,
        value: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        NdArrayMathOps::scatter(dim, tensor, indices, value)
    }

    fn float_select(
        tensor: NdArrayTensor<E>,
        dim: usize,
        indices: NdArrayTensor<i64>,
    ) -> NdArrayTensor<E> {
        NdArrayMathOps::select(tensor, dim, indices)
    }

    fn float_select_assign(
        tensor: NdArrayTensor<E>,
        dim: usize,
        indices: NdArrayTensor<i64>,
        value: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        NdArrayMathOps::select_assign(tensor, dim, indices, value)
    }

    fn float_slice(tensor: NdArrayTensor<E>, ranges: &[Range<usize>]) -> NdArrayTensor<E> {
        NdArrayOps::slice(tensor, ranges)
    }

    fn float_slice_assign(
        tensor: NdArrayTensor<E>,
        ranges: &[Range<usize>],
        value: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        NdArrayOps::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where(
        tensor: NdArrayTensor<E>,
        mask: NdArrayTensor<bool>,
        value: NdArrayTensor<E>,
    ) -> NdArrayTensor<E> {
        NdArrayMathOps::mask_where(tensor, mask, value)
    }

    fn float_mask_fill(
        tensor: NdArrayTensor<E>,
        mask: NdArrayTensor<bool>,
        value: E,
    ) -> NdArrayTensor<E> {
        NdArrayMathOps::mask_fill(tensor, mask, value)
    }

    fn float_equal(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<bool> {
        let output = Zip::from(&lhs.array)
            .and(&rhs.array)
            .map_collect(|&lhs_val, &rhs_val| (lhs_val == rhs_val))
            .into_shared();
        NdArrayTensor::new(output)
    }

    fn float_equal_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a == rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_greater(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<bool> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_greater_elem(tensor, zero)
    }

    fn float_greater_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a > rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_greater_equal(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<bool> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_greater_equal_elem(tensor, zero)
    }

    fn float_greater_equal_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a >= rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_lower(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<bool> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_lower_elem(tensor, zero)
    }

    fn float_lower_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a < rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_lower_equal(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<bool> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_lower_equal_elem(tensor, zero)
    }

    fn float_lower_equal_elem(lhs: NdArrayTensor<E>, rhs: E) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a <= rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_detach(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        tensor
    }

    fn float_mean(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        NdArrayMathOps::mean(tensor)
    }

    fn float_sum(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        NdArrayMathOps::sum(tensor)
    }

    fn float_mean_dim(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<E> {
        NdArrayMathOps::mean_dim(tensor, dim)
    }

    fn float_sum_dim(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<E> {
        NdArrayMathOps::sum_dim(tensor, dim)
    }

    fn float_argmax(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<i64> {
        NdArrayMathOps::argmax(tensor, dim)
    }

    fn float_argmin(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<i64> {
        NdArrayMathOps::argmin(tensor, dim)
    }

    fn float_exp(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor.array.mapv_into(|a| a.exp_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_log(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor.array.mapv_into(|a| a.log_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_log1p(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor.array.mapv_into(|a| a.log1p_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_powf_scalar(tensor: NdArrayTensor<E>, value: f32) -> NdArrayTensor<E> {
        let array = if value == 2.0 {
            // Happens often and is faster.
            tensor.array.mapv_into(|a| a * a).into_shared()
        } else if value.floor() == value {
            // Is faster then powf
            tensor
                .array
                .mapv_into(|a| a.powi_elem(value as i32))
                .into_shared()
        } else {
            // Default
            tensor.array.mapv_into(|a| a.powf_elem(value)).into_shared()
        };

        NdArrayTensor::new(array)
    }

    fn float_sqrt(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor.array.mapv_into(|a| a.sqrt_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_abs(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor.array.mapv_into(|a| a.abs_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_cos(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor
            .array
            .mapv_into(|a| (a.to_f64()).cos().elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn float_sin(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor
            .array
            .mapv_into(|a| (a.to_f64()).sin().elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn float_tanh(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor
            .array
            .mapv_into(|a| (a.to_f64()).tanh().elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn float_erf(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        let array = tensor
            .array
            .mapv_into(|a| erf(a.to_f64()).elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn float_cat(tensors: Vec<NdArrayTensor<E>>, dim: usize) -> NdArrayTensor<E> {
        NdArrayOps::cat(tensors, dim)
    }

    fn float_clamp_min(tensor: NdArrayTensor<E>, min: E) -> NdArrayTensor<E> {
        NdArrayMathOps::clamp_min(tensor, min)
    }

    fn float_clamp_max(tensor: NdArrayTensor<E>, max: E) -> NdArrayTensor<E> {
        NdArrayMathOps::clamp_max(tensor, max)
    }

    fn float_clamp(tensor: NdArrayTensor<E>, min: E, max: E) -> NdArrayTensor<E> {
        NdArrayMathOps::clamp(tensor, min, max)
    }

    fn float_into_int(tensor: NdArrayTensor<E>) -> <NdArray<E> as Backend>::IntTensorPrimitive {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();
        NdArrayTensor { array }
    }

    fn float_powf(lhs: NdArrayTensor<E>, rhs: NdArrayTensor<E>) -> NdArrayTensor<E> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a, b| a.powf_elem(b.to_f32()))
    }

    fn float_permute(tensor: NdArrayTensor<E>, axes: &[usize]) -> NdArrayTensor<E> {
        NdArrayOps::permute(tensor, axes)
    }

    fn float_flip(tensor: NdArrayTensor<E>, axes: &[usize]) -> NdArrayTensor<E> {
        NdArrayOps::flip(tensor, axes)
    }

    fn float_sign(tensor: NdArrayTensor<E>) -> NdArrayTensor<E> {
        NdArrayMathOps::sign_op(tensor)
    }

    fn float_expand(tensor: NdArrayTensor<E>, shape: Shape) -> NdArrayTensor<E> {
        NdArrayOps::expand(tensor, shape)
    }
}
