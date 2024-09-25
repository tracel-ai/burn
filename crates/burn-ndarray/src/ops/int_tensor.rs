// Language
use alloc::vec;
use alloc::vec::Vec;
use burn_common::rand::get_seeded_rng;
use burn_tensor::ops::IntTensorOps;
use burn_tensor::Distribution;

use burn_tensor::ElementConversion;
use core::ops::Range;
use ndarray::IntoDimension;
use ndarray::Zip;

// Current crate
use crate::element::ExpElement;
use crate::element::FloatNdArrayElement;
use crate::element::QuantElement;
use crate::{tensor::NdArrayTensor, NdArray};
use crate::{NdArrayDevice, SEED};

// Workspace crates
use burn_tensor::{backend::Backend, Shape, TensorData};

use super::{NdArrayMathOps, NdArrayOps};

impl<E: FloatNdArrayElement, Q: QuantElement> IntTensorOps<Self> for NdArray<E, Q> {
    fn int_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor<i64> {
        NdArrayTensor::from_data(data)
    }

    fn int_shape(tensor: &NdArrayTensor<i64>) -> Shape {
        tensor.shape()
    }

    async fn int_into_data(tensor: NdArrayTensor<i64>) -> TensorData {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        TensorData::new(values, shape)
    }

    fn int_to_device(tensor: NdArrayTensor<i64>, _device: &NdArrayDevice) -> NdArrayTensor<i64> {
        tensor
    }

    fn int_reshape(tensor: NdArrayTensor<i64>, shape: Shape) -> NdArrayTensor<i64> {
        NdArrayOps::reshape(tensor, shape)
    }

    fn int_slice(tensor: NdArrayTensor<i64>, ranges: &[Range<usize>]) -> NdArrayTensor<i64> {
        NdArrayOps::slice(tensor, ranges)
    }

    fn int_device(_tensor: &NdArrayTensor<i64>) -> <NdArray<E> as Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn int_empty(shape: Shape, _device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<i64> {
        let values = vec![0; shape.num_elements()];
        NdArrayTensor::from_data(TensorData::new(values, shape))
    }

    fn int_mask_where(
        tensor: NdArrayTensor<i64>,
        mask: NdArrayTensor<bool>,
        source: NdArrayTensor<i64>,
    ) -> NdArrayTensor<i64> {
        NdArrayMathOps::mask_where(tensor, mask, source)
    }

    fn int_mask_fill(
        tensor: NdArrayTensor<i64>,
        mask: NdArrayTensor<bool>,
        value: i64,
    ) -> NdArrayTensor<i64> {
        NdArrayMathOps::mask_fill(tensor, mask, value)
    }

    fn int_slice_assign(
        tensor: NdArrayTensor<i64>,
        ranges: &[Range<usize>],
        value: NdArrayTensor<i64>,
    ) -> NdArrayTensor<i64> {
        NdArrayOps::slice_assign(tensor, ranges, value)
    }

    fn int_cat(tensors: Vec<NdArrayTensor<i64>>, dim: usize) -> NdArrayTensor<i64> {
        NdArrayOps::cat(tensors, dim)
    }

    fn int_equal(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<bool> {
        let output = Zip::from(&lhs.array)
            .and(&rhs.array)
            .map_collect(|&lhs_val, &rhs_val| (lhs_val == rhs_val))
            .into_shared();
        NdArrayTensor::new(output)
    }

    fn int_equal_elem(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a == rhs).into_shared();
        NdArrayTensor { array }
    }

    fn int_greater(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<bool> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_greater_elem(tensor, 0)
    }

    fn int_greater_elem(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a > rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_greater_equal(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<bool> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_greater_equal_elem(tensor, 0)
    }

    fn int_greater_equal_elem(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a >= rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_lower(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<bool> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_lower_elem(tensor, 0)
    }

    fn int_lower_elem(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a < rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_lower_equal(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<bool> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_lower_equal_elem(tensor, 0)
    }

    fn int_lower_equal_elem(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a <= rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_add(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        NdArrayMathOps::add(lhs, rhs)
    }

    fn int_add_scalar(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<i64> {
        NdArrayMathOps::add_scalar(lhs, rhs)
    }

    fn int_sub(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        NdArrayMathOps::sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<i64> {
        NdArrayMathOps::sub_scalar(lhs, rhs)
    }

    fn int_mul(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        NdArrayMathOps::mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<i64> {
        NdArrayMathOps::mul_scalar(lhs, rhs)
    }

    fn int_div(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        NdArrayMathOps::div(lhs, rhs)
    }

    fn int_div_scalar(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<i64> {
        NdArrayMathOps::div_scalar(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: NdArrayTensor<i64>, rhs: i64) -> NdArrayTensor<i64> {
        NdArrayMathOps::remainder_scalar(lhs, rhs)
    }

    fn int_neg(tensor: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        Self::int_mul_scalar(tensor, -1)
    }

    fn int_zeros(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<i64> {
        Self::int_from_data(TensorData::zeros::<i64, _>(shape), device)
    }

    fn int_ones(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<i64> {
        Self::int_from_data(TensorData::ones::<i64, _>(shape), device)
    }

    fn int_full(
        shape: Shape,
        fill_value: i64,
        device: &<NdArray<E> as Backend>::Device,
    ) -> NdArrayTensor<i64> {
        Self::int_from_data(TensorData::full(shape, fill_value), device)
    }

    fn int_sum(tensor: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        NdArrayMathOps::sum(tensor)
    }

    fn int_sum_dim(tensor: NdArrayTensor<i64>, dim: usize) -> NdArrayTensor<i64> {
        NdArrayMathOps::sum_dim(tensor, dim)
    }

    fn int_prod(tensor: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        NdArrayMathOps::prod(tensor)
    }

    fn int_prod_dim(tensor: NdArrayTensor<i64>, dim: usize) -> NdArrayTensor<i64> {
        NdArrayMathOps::prod_dim(tensor, dim)
    }

    fn int_mean(tensor: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        NdArrayMathOps::mean(tensor)
    }

    fn int_mean_dim(tensor: NdArrayTensor<i64>, dim: usize) -> NdArrayTensor<i64> {
        NdArrayMathOps::mean_dim(tensor, dim)
    }

    fn int_gather(
        dim: usize,
        tensor: NdArrayTensor<i64>,
        indices: NdArrayTensor<i64>,
    ) -> NdArrayTensor<i64> {
        NdArrayMathOps::gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: NdArrayTensor<i64>,
        indices: NdArrayTensor<i64>,
        value: NdArrayTensor<i64>,
    ) -> NdArrayTensor<i64> {
        NdArrayMathOps::scatter(dim, tensor, indices, value)
    }

    fn int_select(
        tensor: NdArrayTensor<i64>,
        dim: usize,
        indices: NdArrayTensor<i64>,
    ) -> NdArrayTensor<i64> {
        NdArrayMathOps::select(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: NdArrayTensor<i64>,
        dim: usize,
        indices: NdArrayTensor<i64>,
        value: NdArrayTensor<i64>,
    ) -> NdArrayTensor<i64> {
        NdArrayMathOps::select_assign(tensor, dim, indices, value)
    }
    fn int_argmax(tensor: NdArrayTensor<i64>, dim: usize) -> NdArrayTensor<i64> {
        NdArrayMathOps::argmax(tensor, dim)
    }

    fn int_argmin(tensor: NdArrayTensor<i64>, dim: usize) -> NdArrayTensor<i64> {
        NdArrayMathOps::argmin(tensor, dim)
    }

    fn int_clamp_min(tensor: NdArrayTensor<i64>, min: i64) -> NdArrayTensor<i64> {
        NdArrayMathOps::clamp_min(tensor, min)
    }

    fn int_clamp_max(tensor: NdArrayTensor<i64>, max: i64) -> NdArrayTensor<i64> {
        NdArrayMathOps::clamp_max(tensor, max)
    }

    fn int_clamp(tensor: NdArrayTensor<i64>, min: i64, max: i64) -> NdArrayTensor<i64> {
        NdArrayMathOps::clamp(tensor, min, max)
    }

    fn int_abs(tensor: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        let array = tensor.array.mapv_into(|a| a.int_abs_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn int_into_float(tensor: NdArrayTensor<i64>) -> <NdArray<E> as Backend>::FloatTensorPrimitive {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();
        NdArrayTensor { array }
    }

    fn int_swap_dims(tensor: NdArrayTensor<i64>, dim1: usize, dim2: usize) -> NdArrayTensor<i64> {
        NdArrayOps::swap_dims(tensor, dim1, dim2)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> NdArrayTensor<i64> {
        let mut seed = SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            get_seeded_rng()
        };

        let effective_distribution = if distribution == Distribution::Default {
            Distribution::Uniform(0.0, 255.0) // Assuming UniformInt is the integer variant
        } else {
            distribution
        };

        let tensor = Self::int_from_data(
            TensorData::random::<i64, _, _>(shape, effective_distribution, &mut rng),
            device,
        );
        *seed = Some(rng);
        tensor
    }

    fn int_powi(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &i64, b: &i64| a.pow(*b as u32))
    }

    fn int_powf(lhs: NdArrayTensor<i64>, rhs: NdArrayTensor<E>) -> NdArrayTensor<i64> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &i64, b: &E| a.pow(b.elem::<u32>()))
    }

    fn int_powf_scalar(lhs: NdArrayTensor<i64>, rhs: f32) -> NdArrayTensor<i64> {
        NdArrayMathOps::elementwise_op_scalar(lhs, |a: i64| a.pow(rhs as u32))
    }

    fn int_permute(tensor: NdArrayTensor<i64>, axes: &[usize]) -> NdArrayTensor<i64> {
        let array = tensor.array.permuted_axes(axes.into_dimension());
        NdArrayTensor { array }
    }

    fn int_flip(tensor: NdArrayTensor<i64>, axes: &[usize]) -> NdArrayTensor<i64> {
        NdArrayOps::flip(tensor, axes)
    }

    fn int_sign(tensor: NdArrayTensor<i64>) -> NdArrayTensor<i64> {
        NdArrayMathOps::sign_op(tensor)
    }

    fn int_expand(tensor: NdArrayTensor<i64>, shape: Shape) -> NdArrayTensor<i64> {
        NdArrayOps::expand(tensor, shape)
    }
}
