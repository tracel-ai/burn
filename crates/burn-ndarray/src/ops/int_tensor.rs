// Language
use alloc::vec::Vec;
use burn_common::rand::get_seeded_rng;
use burn_tensor::ops::FloatTensor;
use burn_tensor::ops::IntTensorOps;
use burn_tensor::Distribution;

use burn_tensor::ElementConversion;
use core::ops::Range;
use ndarray::IntoDimension;
use ndarray::Zip;

// Current crate
use crate::element::FloatNdArrayElement;
use crate::element::IntNdArrayElement;
use crate::element::QuantElement;
use crate::execute_with_float_dtype;
use crate::new_tensor_float;
use crate::{tensor::NdArrayTensor, NdArray};
use crate::{NdArrayDevice, SEED};

// Workspace crates
use burn_tensor::{backend::Backend, DType, Shape, TensorData};

use super::{NdArrayMathOps, NdArrayOps};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> IntTensorOps<Self>
    for NdArray<E, I, Q>
{
    fn int_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor<I> {
        match data.dtype {
            DType::I64 | DType::I32 => NdArrayTensor::from_data(data),
            _ => unimplemented!("Unsupported dtype for `int_from_data`"),
        }
    }

    async fn int_into_data(tensor: NdArrayTensor<I>) -> TensorData {
        NdArrayOps::into_data(tensor)
    }

    fn int_to_device(tensor: NdArrayTensor<I>, _device: &NdArrayDevice) -> NdArrayTensor<I> {
        tensor
    }

    fn int_reshape(tensor: NdArrayTensor<I>, shape: Shape) -> NdArrayTensor<I> {
        NdArrayOps::reshape(tensor, shape)
    }

    fn int_slice(tensor: NdArrayTensor<I>, ranges: &[Range<usize>]) -> NdArrayTensor<I> {
        NdArrayOps::slice(tensor, ranges)
    }

    fn int_device(_tensor: &NdArrayTensor<I>) -> <NdArray<E> as Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn int_empty(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<I> {
        Self::int_zeros(shape, device)
    }

    fn int_mask_where(
        tensor: NdArrayTensor<I>,
        mask: NdArrayTensor<bool>,
        source: NdArrayTensor<I>,
    ) -> NdArrayTensor<I> {
        NdArrayMathOps::mask_where(tensor, mask, source)
    }

    fn int_mask_fill(
        tensor: NdArrayTensor<I>,
        mask: NdArrayTensor<bool>,
        value: I,
    ) -> NdArrayTensor<I> {
        NdArrayMathOps::mask_fill(tensor, mask, value)
    }

    fn int_slice_assign(
        tensor: NdArrayTensor<I>,
        ranges: &[Range<usize>],
        value: NdArrayTensor<I>,
    ) -> NdArrayTensor<I> {
        NdArrayOps::slice_assign(tensor, ranges, value)
    }

    fn int_cat(tensors: Vec<NdArrayTensor<I>>, dim: usize) -> NdArrayTensor<I> {
        NdArrayOps::cat(tensors, dim)
    }

    fn int_equal(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<bool> {
        let output = Zip::from(&lhs.array)
            .and(&rhs.array)
            .map_collect(|&lhs_val, &rhs_val| (lhs_val == rhs_val))
            .into_shared();
        NdArrayTensor::new(output)
    }

    fn int_equal_elem(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a == rhs).into_shared();
        NdArrayTensor { array }
    }

    fn int_greater(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<bool> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_greater_elem(tensor, 0.elem())
    }

    fn int_greater_elem(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a > rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_greater_equal(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<bool> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_greater_equal_elem(tensor, 0.elem())
    }

    fn int_greater_equal_elem(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a >= rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_lower(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<bool> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_lower_elem(tensor, 0.elem())
    }

    fn int_lower_elem(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a < rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_lower_equal(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<bool> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_lower_equal_elem(tensor, 0.elem())
    }

    fn int_lower_equal_elem(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<bool> {
        let array = lhs.array.mapv(|a| a <= rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_add(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::add(lhs, rhs)
    }

    fn int_add_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        NdArrayMathOps::add_scalar(lhs, rhs)
    }

    fn int_sub(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::sub(lhs, rhs)
    }

    fn int_sub_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        NdArrayMathOps::sub_scalar(lhs, rhs)
    }

    fn int_mul(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::mul(lhs, rhs)
    }

    fn int_mul_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        NdArrayMathOps::mul_scalar(lhs, rhs)
    }

    fn int_div(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::div(lhs, rhs)
    }

    fn int_div_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        NdArrayMathOps::div_scalar(lhs, rhs)
    }

    fn int_remainder(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::remainder(lhs, rhs)
    }

    fn int_remainder_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        NdArrayMathOps::remainder_scalar(lhs, rhs)
    }

    fn int_neg(tensor: NdArrayTensor<I>) -> NdArrayTensor<I> {
        Self::int_mul_scalar(tensor, (-1).elem())
    }

    fn int_zeros(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<I> {
        Self::int_from_data(TensorData::zeros::<I, _>(shape), device)
    }

    fn int_ones(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<I> {
        Self::int_from_data(TensorData::ones::<I, _>(shape), device)
    }

    fn int_full(
        shape: Shape,
        fill_value: I,
        device: &<NdArray<E> as Backend>::Device,
    ) -> NdArrayTensor<I> {
        Self::int_from_data(TensorData::full(shape, fill_value), device)
    }

    fn int_sum(tensor: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::sum(tensor)
    }

    fn int_sum_dim(tensor: NdArrayTensor<I>, dim: usize) -> NdArrayTensor<I> {
        NdArrayMathOps::sum_dim(tensor, dim)
    }

    fn int_prod(tensor: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::prod(tensor)
    }

    fn int_prod_dim(tensor: NdArrayTensor<I>, dim: usize) -> NdArrayTensor<I> {
        NdArrayMathOps::prod_dim(tensor, dim)
    }

    fn int_mean(tensor: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::mean(tensor)
    }

    fn int_mean_dim(tensor: NdArrayTensor<I>, dim: usize) -> NdArrayTensor<I> {
        NdArrayMathOps::mean_dim(tensor, dim)
    }

    fn int_gather(
        dim: usize,
        tensor: NdArrayTensor<I>,
        indices: NdArrayTensor<I>,
    ) -> NdArrayTensor<I> {
        NdArrayMathOps::gather(dim, tensor, indices)
    }

    fn int_scatter(
        dim: usize,
        tensor: NdArrayTensor<I>,
        indices: NdArrayTensor<I>,
        value: NdArrayTensor<I>,
    ) -> NdArrayTensor<I> {
        NdArrayMathOps::scatter(dim, tensor, indices, value)
    }

    fn int_select(
        tensor: NdArrayTensor<I>,
        dim: usize,
        indices: NdArrayTensor<I>,
    ) -> NdArrayTensor<I> {
        NdArrayMathOps::select(tensor, dim, indices)
    }

    fn int_select_assign(
        tensor: NdArrayTensor<I>,
        dim: usize,
        indices: NdArrayTensor<I>,
        value: NdArrayTensor<I>,
    ) -> NdArrayTensor<I> {
        NdArrayMathOps::select_assign(tensor, dim, indices, value)
    }
    fn int_argmax(tensor: NdArrayTensor<I>, dim: usize) -> NdArrayTensor<I> {
        NdArrayMathOps::argmax(tensor, dim)
    }

    fn int_argmin(tensor: NdArrayTensor<I>, dim: usize) -> NdArrayTensor<I> {
        NdArrayMathOps::argmin(tensor, dim)
    }

    fn int_clamp_min(tensor: NdArrayTensor<I>, min: I) -> NdArrayTensor<I> {
        NdArrayMathOps::clamp_min(tensor, min)
    }

    fn int_clamp_max(tensor: NdArrayTensor<I>, max: I) -> NdArrayTensor<I> {
        NdArrayMathOps::clamp_max(tensor, max)
    }

    fn int_clamp(tensor: NdArrayTensor<I>, min: I, max: I) -> NdArrayTensor<I> {
        NdArrayMathOps::clamp(tensor, min, max)
    }

    fn int_abs(tensor: NdArrayTensor<I>) -> NdArrayTensor<I> {
        let array = tensor.array.mapv_into(|a| a.int_abs_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn int_into_float(tensor: NdArrayTensor<I>) -> FloatTensor<Self> {
        new_tensor_float!(NdArrayTensor {
            array: tensor.array.mapv(|a| a.elem()).into_shared()
        })
    }

    fn int_swap_dims(tensor: NdArrayTensor<I>, dim1: usize, dim2: usize) -> NdArrayTensor<I> {
        NdArrayOps::swap_dims(tensor, dim1, dim2)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> NdArrayTensor<I> {
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
            TensorData::random::<I, _, _>(shape, effective_distribution, &mut rng),
            device,
        );
        *seed = Some(rng);
        tensor
    }

    fn int_powi(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>().pow(b.elem::<u32>())).elem()
        })
    }

    fn int_powf(lhs: NdArrayTensor<I>, rhs: FloatTensor<Self>) -> NdArrayTensor<I> {
        execute_with_float_dtype!(rhs => |rhs| {
            NdArrayMathOps::elementwise_op(lhs, rhs, |a, b| {
                (a.elem::<i64>().pow(*b as u32)).elem()
            })
        })
    }

    fn int_powf_scalar(lhs: NdArrayTensor<I>, rhs: f32) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| (a.elem::<i64>().pow(rhs as u32)).elem())
    }

    fn int_permute(tensor: NdArrayTensor<I>, axes: &[usize]) -> NdArrayTensor<I> {
        let array = tensor.array.permuted_axes(axes.into_dimension());
        NdArrayTensor { array }
    }

    fn int_flip(tensor: NdArrayTensor<I>, axes: &[usize]) -> NdArrayTensor<I> {
        NdArrayOps::flip(tensor, axes)
    }

    fn int_sign(tensor: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::sign_op(tensor)
    }

    fn int_expand(tensor: NdArrayTensor<I>, shape: Shape) -> NdArrayTensor<I> {
        NdArrayOps::expand(tensor, shape)
    }

    fn bitwise_and(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() & (b.elem::<i64>())).elem()
        })
    }

    fn bitwise_and_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
            (a.elem::<i64>() & rhs.elem::<i64>()).elem()
        })
    }

    fn bitwise_or(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() | (b.elem::<i64>())).elem()
        })
    }

    fn bitwise_or_scalar(
        lhs: burn_tensor::ops::IntTensor<Self>,
        rhs: burn_tensor::ops::IntElem<Self>,
    ) -> burn_tensor::ops::IntTensor<Self> {
        NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
            (a.elem::<i64>() | rhs.elem::<i64>()).elem()
        })
    }

    fn bitwise_xor(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() ^ (b.elem::<i64>())).elem()
        })
    }

    fn bitwise_xor_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
            (a.elem::<i64>() ^ rhs.elem::<i64>()).elem()
        })
    }

    fn bitwise_not(tensor: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op_scalar(tensor, |a: I| (!a.elem::<i64>()).elem())
    }

    fn bitwise_left_shift(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() << (b.elem::<u32>())).elem()
        })
    }

    fn bitwise_left_shift_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
            (a.elem::<i64>() << rhs.elem::<u32>()).elem()
        })
    }

    fn bitwise_right_shift(lhs: NdArrayTensor<I>, rhs: NdArrayTensor<I>) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
            (a.elem::<i64>() >> (b.elem::<u32>())).elem()
        })
    }

    fn bitwise_right_shift_scalar(lhs: NdArrayTensor<I>, rhs: I) -> NdArrayTensor<I> {
        NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
            (a.elem::<i64>() >> rhs.elem::<u32>()).elem()
        })
    }
}
