// Language
use alloc::vec;
use alloc::vec::Vec;
use burn_common::rand::get_seeded_rng;
use burn_tensor::ops::ByteTensorOps;
use burn_tensor::Distribution;

use burn_tensor::ElementConversion;
use burn_tensor::TensorMetadata;
use core::ops::Range;
use ndarray::IntoDimension;
use ndarray::Zip;

// Current crate
use crate::element::IntNdArrayElement;
use crate::element::QuantElement;
use crate::element::{ExpElement, FloatNdArrayElement};
use crate::{tensor::NdArrayTensor, NdArray};
use crate::{NdArrayDevice, SEED};

// Workspace crates
use burn_tensor::{backend::Backend, Shape, TensorData};

use super::{NdArrayMathOps, NdArrayOps};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ByteTensorOps<Self>
    for NdArray<E, I, Q>
{
    fn byte_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor<u8> {
        NdArrayTensor::from_data(data)
    }

    async fn byte_into_data(tensor: NdArrayTensor<u8>) -> TensorData {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        TensorData::new(values, shape)
    }

    fn byte_to_device(tensor: NdArrayTensor<u8>, _device: &NdArrayDevice) -> NdArrayTensor<u8> {
        tensor
    }

    fn byte_reshape(tensor: NdArrayTensor<u8>, shape: Shape) -> NdArrayTensor<u8> {
        NdArrayOps::reshape(tensor, shape)
    }

    fn byte_slice(tensor: NdArrayTensor<u8>, ranges: &[Range<usize>]) -> NdArrayTensor<u8> {
        NdArrayOps::slice(tensor, ranges)
    }

    fn byte_device(_tensor: &NdArrayTensor<u8>) -> <NdArray<E> as Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn byte_empty(shape: Shape, _device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<u8> {
        let values = vec![0; shape.num_elements()];
        NdArrayTensor::from_data(TensorData::new(values, shape))
    }

    fn byte_mask_where(
        tensor: NdArrayTensor<u8>,
        mask: NdArrayTensor<bool>,
        source: NdArrayTensor<u8>,
    ) -> NdArrayTensor<u8> {
        NdArrayMathOps::mask_where(tensor, mask, source)
    }

    fn byte_mask_fill(
        tensor: NdArrayTensor<u8>,
        mask: NdArrayTensor<bool>,
        value: u8,
    ) -> NdArrayTensor<u8> {
        NdArrayMathOps::mask_fill(tensor, mask, value)
    }

    fn byte_slice_assign(
        tensor: NdArrayTensor<u8>,
        ranges: &[Range<usize>],
        value: NdArrayTensor<u8>,
    ) -> NdArrayTensor<u8> {
        NdArrayOps::slice_assign(tensor, ranges, value)
    }

    fn byte_cat(tensors: Vec<NdArrayTensor<u8>>, dim: usize) -> NdArrayTensor<u8> {
        NdArrayOps::cat(tensors, dim)
    }

    fn byte_equal(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<bool> {
        let output = Zip::from(&lhs.array)
            .and(&rhs.array)
            .map_collect(|&lhs_val, &rhs_val| (lhs_val == rhs_val))
            .into_shared();
        NdArrayTensor::new(output)
    }

    fn byte_equal_elem(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a == rhs).into_shared();
        NdArrayTensor { array }
    }

    fn byte_greater(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<bool> {
        let tensor = Self::byte_sub(lhs, rhs);
        Self::byte_greater_elem(tensor, 0.elem())
    }

    fn byte_greater_elem(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a > rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn byte_greater_equal(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<bool> {
        let tensor = Self::byte_sub(lhs, rhs);
        Self::byte_greater_equal_elem(tensor, 0.elem())
    }

    fn byte_greater_equal_elem(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a >= rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn byte_lower(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<bool> {
        let tensor = Self::byte_sub(lhs, rhs);
        Self::byte_lower_elem(tensor, 0.elem())
    }

    fn byte_lower_elem(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a < rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn byte_lower_equal(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<bool> {
        let tensor = Self::byte_sub(lhs, rhs);
        Self::byte_lower_equal_elem(tensor, 0.elem())
    }

    fn byte_lower_equal_elem(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a <= rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn byte_add(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        NdArrayMathOps::add(lhs, rhs)
    }

    fn byte_add_scalar(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<u8> {
        NdArrayMathOps::add_scalar(lhs, rhs)
    }

    fn byte_sub(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        NdArrayMathOps::sub(lhs, rhs)
    }

    fn byte_sub_scalar(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<u8> {
        NdArrayMathOps::sub_scalar(lhs, rhs)
    }

    fn byte_mul(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        NdArrayMathOps::mul(lhs, rhs)
    }

    fn byte_mul_scalar(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<u8> {
        NdArrayMathOps::mul_scalar(lhs, rhs)
    }

    fn byte_div(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        NdArrayMathOps::div(lhs, rhs)
    }

    fn byte_div_scalar(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<u8> {
        NdArrayMathOps::div_scalar(lhs, rhs)
    }

    fn byte_remainder(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        NdArrayMathOps::remainder(lhs, rhs)
    }

    fn byte_remainder_scalar(lhs: NdArrayTensor<u8>, rhs: u8) -> NdArrayTensor<u8> {
        NdArrayMathOps::remainder_scalar(lhs, rhs)
    }

    fn byte_neg(tensor: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        Self::byte_mul_scalar(tensor, (-1).elem())
    }

    fn byte_zeros(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<u8> {
        Self::byte_from_data(TensorData::zeros::<i64, _>(shape), device)
    }

    fn byte_ones(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<u8> {
        Self::byte_from_data(TensorData::ones::<i64, _>(shape), device)
    }

    fn byte_full(
        shape: Shape,
        fill_value: u8,
        device: &<NdArray<E> as Backend>::Device,
    ) -> NdArrayTensor<u8> {
        Self::byte_from_data(TensorData::full(shape, fill_value), device)
    }

    fn byte_sum(tensor: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        NdArrayMathOps::sum(tensor)
    }

    fn byte_sum_dim(tensor: NdArrayTensor<u8>, dim: usize) -> NdArrayTensor<u8> {
        NdArrayMathOps::sum_dim(tensor, dim)
    }

    fn byte_prod(tensor: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        NdArrayMathOps::prod(tensor)
    }

    fn byte_prod_dim(tensor: NdArrayTensor<u8>, dim: usize) -> NdArrayTensor<u8> {
        NdArrayMathOps::prod_dim(tensor, dim)
    }

    fn byte_mean(tensor: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        NdArrayMathOps::mean(tensor)
    }

    fn byte_mean_dim(tensor: NdArrayTensor<u8>, dim: usize) -> NdArrayTensor<u8> {
        NdArrayMathOps::mean_dim(tensor, dim)
    }

    fn byte_gather(
        dim: usize,
        tensor: NdArrayTensor<u8>,
        indices: NdArrayTensor<I>,
    ) -> NdArrayTensor<u8> {
        NdArrayMathOps::gather(dim, tensor, indices)
    }

    fn byte_scatter(
        dim: usize,
        tensor: NdArrayTensor<u8>,
        indices: NdArrayTensor<I>,
        value: NdArrayTensor<u8>,
    ) -> NdArrayTensor<u8> {
        NdArrayMathOps::scatter(dim, tensor, indices, value)
    }

    fn byte_select(
        tensor: NdArrayTensor<u8>,
        dim: usize,
        indices: NdArrayTensor<I>,
    ) -> NdArrayTensor<u8> {
        NdArrayMathOps::select(tensor, dim, indices)
    }

    fn byte_select_assign(
        tensor: NdArrayTensor<u8>,
        dim: usize,
        indices: NdArrayTensor<I>,
        value: NdArrayTensor<u8>,
    ) -> NdArrayTensor<u8> {
        NdArrayMathOps::select_assign(tensor, dim, indices, value)
    }
    fn byte_argmax(tensor: NdArrayTensor<u8>, dim: usize) -> NdArrayTensor<I> {
        NdArrayMathOps::argmax(tensor, dim)
    }

    fn byte_argmin(tensor: NdArrayTensor<u8>, dim: usize) -> NdArrayTensor<I> {
        NdArrayMathOps::argmin(tensor, dim)
    }

    fn byte_clamp_min(tensor: NdArrayTensor<u8>, min: u8) -> NdArrayTensor<u8> {
        NdArrayMathOps::clamp_min(tensor, min)
    }

    fn byte_clamp_max(tensor: NdArrayTensor<u8>, max: u8) -> NdArrayTensor<u8> {
        NdArrayMathOps::clamp_max(tensor, max)
    }

    fn byte_clamp(tensor: NdArrayTensor<u8>, min: u8, max: u8) -> NdArrayTensor<u8> {
        NdArrayMathOps::clamp(tensor, min, max)
    }

    fn byte_abs(tensor: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        let array = tensor.array.mapv_into(|a| a.abs_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn byte_into_float(tensor: NdArrayTensor<u8>) -> <NdArray<E> as Backend>::FloatTensorPrimitive {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();
        NdArrayTensor { array }
    }

    fn byte_into_int(
        tensor: NdArrayTensor<u8>,
    ) -> <NdArray<E, I, Q> as Backend>::IntTensorPrimitive {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();
        NdArrayTensor { array }
    }

    fn byte_swap_dims(tensor: NdArrayTensor<u8>, dim1: usize, dim2: usize) -> NdArrayTensor<u8> {
        NdArrayOps::swap_dims(tensor, dim1, dim2)
    }

    fn byte_random(
        shape: Shape,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> NdArrayTensor<u8> {
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

        let tensor = Self::byte_from_data(
            TensorData::random::<i64, _, _>(shape, effective_distribution, &mut rng),
            device,
        );
        *seed = Some(rng);
        tensor
    }

    fn byte_powi(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &u8, b: &u8| {
            (a.elem::<i64>().pow(b.elem::<u32>())).elem()
        })
    }

    fn byte_powf(lhs: NdArrayTensor<u8>, rhs: NdArrayTensor<E>) -> NdArrayTensor<u8> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &u8, b: &E| {
            (a.elem::<i64>().pow(b.elem::<u32>())).elem()
        })
    }

    fn byte_powf_scalar(lhs: NdArrayTensor<u8>, rhs: f32) -> NdArrayTensor<u8> {
        NdArrayMathOps::elementwise_op_scalar(lhs, |a: u8| (a.elem::<i64>().pow(rhs as u32)).elem())
    }

    fn byte_permute(tensor: NdArrayTensor<u8>, axes: &[usize]) -> NdArrayTensor<u8> {
        let array = tensor.array.permuted_axes(axes.into_dimension());
        NdArrayTensor { array }
    }

    fn byte_flip(tensor: NdArrayTensor<u8>, axes: &[usize]) -> NdArrayTensor<u8> {
        NdArrayOps::flip(tensor, axes)
    }

    fn byte_sign(tensor: NdArrayTensor<u8>) -> NdArrayTensor<u8> {
        Self::byte_zeros(tensor.shape(), &Self::byte_device(&tensor))
    }

    fn byte_expand(tensor: NdArrayTensor<u8>, shape: Shape) -> NdArrayTensor<u8> {
        NdArrayOps::expand(tensor, shape)
    }
}
