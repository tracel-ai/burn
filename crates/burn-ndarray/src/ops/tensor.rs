// Language
use alloc::vec::Vec;
use core::ops::Range;
use ndarray::IntoDimension;

// Current crate
use super::{matmul::matmul, NdArrayMathOps, NdArrayOps};
use crate::element::FloatNdArrayElement;
use crate::{tensor::NdArrayTensor, NdArray};
use crate::{NdArrayDevice, SEED};

// Workspace crates
use burn_common::rand::get_seeded_rng;
use burn_tensor::{backend::Backend, ops::FloatTensorOps, Data, ElementConversion, Shape};
use burn_tensor::{Distribution, Reader};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use libm::erf;

impl<E: FloatNdArrayElement> FloatTensorOps<Self> for NdArray<E> {
    fn float_from_data<const D: usize>(
        data: Data<E, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<E, D> {
        NdArrayTensor::from_data(data)
    }

    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> NdArrayTensor<E, D> {
        let mut seed = SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            get_seeded_rng()
        };
        let tensor = Self::float_from_data(Data::random(shape, distribution, &mut rng), device);
        *seed = Some(rng);
        tensor
    }

    fn float_shape<const D: usize>(tensor: &NdArrayTensor<E, D>) -> Shape<D> {
        tensor.shape()
    }

    fn float_into_data<const D: usize>(
        tensor: NdArrayTensor<E, D>,
    ) -> Reader<Data<<NdArray<E> as Backend>::FloatElem, D>> {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();

        Reader::Concrete(Data::new(values, shape))
    }

    fn float_device<const D: usize>(_tensor: &NdArrayTensor<E, D>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn float_to_device<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<E, D> {
        tensor
    }

    fn float_empty<const D: usize>(
        shape: Shape<D>,
        device: &<NdArray<E> as Backend>::Device,
    ) -> NdArrayTensor<E, D> {
        NdArray::<E>::float_zeros(shape, device)
    }

    fn float_add<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::add(lhs, rhs)
    }

    fn float_add_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::add_scalar(lhs, rhs)
    }

    fn float_sub<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::sub(lhs, rhs)
    }

    fn float_sub_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::sub_scalar(lhs, rhs)
    }

    fn float_mul<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mul(lhs, rhs)
    }

    fn float_mul_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mul_scalar(lhs, rhs)
    }

    fn float_div<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::div(lhs, rhs)
    }

    fn float_div_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::div_scalar(lhs, rhs)
    }

    fn float_remainder_scalar<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: E,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::remainder_scalar(lhs, rhs)
    }

    fn float_matmul<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        matmul(lhs, rhs)
    }

    fn float_neg<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        Self::float_mul_scalar(tensor, (-1f32).elem::<E>())
    }

    fn float_recip<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        NdArrayMathOps::recip(tensor)
    }

    fn float_swap_dims<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> NdArrayTensor<E, D> {
        NdArrayOps::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        shape: Shape<D2>,
    ) -> NdArrayTensor<E, D2> {
        NdArrayOps::reshape(tensor, shape)
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: NdArrayTensor<E, D>,
        indices: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::gather(dim, tensor, indices)
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: NdArrayTensor<E, D>,
        indices: NdArrayTensor<i64, D>,
        value: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::scatter(dim, tensor, indices, value)
    }

    fn float_select<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
        indices: NdArrayTensor<i64, 1>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::select(tensor, dim, indices)
    }

    fn float_select_assign<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
        indices: NdArrayTensor<i64, 1>,
        value: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::select_assign(tensor, dim, indices, value)
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        ranges: [Range<usize>; D2],
    ) -> NdArrayTensor<E, D1> {
        NdArrayOps::slice(tensor, ranges)
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        ranges: [Range<usize>; D2],
        value: NdArrayTensor<E, D1>,
    ) -> NdArrayTensor<E, D1> {
        NdArrayOps::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        mask: NdArrayTensor<bool, D>,
        value: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mask_where(tensor, mask, value)
    }

    fn float_mask_fill<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        mask: NdArrayTensor<bool, D>,
        value: E,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mask_fill(tensor, mask, value)
    }

    fn float_equal<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();

        Self::float_equal_elem(tensor, zero)
    }

    fn float_equal_elem<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a == rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_greater<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_greater_elem(tensor, zero)
    }

    fn float_greater_elem<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a > rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_greater_equal<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_greater_equal_elem(tensor, zero)
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a >= rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_lower<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_lower_elem(tensor, zero)
    }

    fn float_lower_elem<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a < rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_lower_equal<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_lower_equal_elem(tensor, zero)
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a <= rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_detach<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        tensor
    }

    fn float_mean<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, 1> {
        NdArrayMathOps::mean(tensor)
    }

    fn float_sum<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, 1> {
        NdArrayMathOps::sum(tensor)
    }

    fn float_mean_dim<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mean_dim(tensor, dim)
    }

    fn float_sum_dim<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::sum_dim(tensor, dim)
    }

    fn float_argmax<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::argmax(tensor, dim)
    }

    fn float_argmin<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::argmin(tensor, dim)
    }

    fn float_exp<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.exp_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_log<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.log_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_log1p<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.log1p_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_powf_scalar<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        value: f32,
    ) -> NdArrayTensor<E, D> {
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

    fn float_sqrt<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.sqrt_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_abs<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.abs_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn float_cos<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv_into(|a| (a.to_f64()).cos().elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn float_sin<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv_into(|a| (a.to_f64()).sin().elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn float_tanh<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv_into(|a| (a.to_f64()).tanh().elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn float_erf<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv_into(|a| erf(a.to_f64()).elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn float_cat<const D: usize>(
        tensors: Vec<NdArrayTensor<E, D>>,
        dim: usize,
    ) -> NdArrayTensor<E, D> {
        NdArrayOps::cat(tensors, dim)
    }

    fn float_clamp_min<const D: usize>(tensor: NdArrayTensor<E, D>, min: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::clamp_min(tensor, min)
    }

    fn float_clamp_max<const D: usize>(tensor: NdArrayTensor<E, D>, max: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::clamp_max(tensor, max)
    }

    fn float_clamp<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        min: E,
        max: E,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::clamp(tensor, min, max)
    }

    fn float_into_int<const D: usize>(
        tensor: <NdArray<E> as Backend>::FloatTensorPrimitive<D>,
    ) -> <NdArray<E> as Backend>::IntTensorPrimitive<D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();
        NdArrayTensor { array }
    }

    fn float_powf<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::elementwise_op(lhs, rhs, |a, b| a.powf_elem(b.to_f32()))
    }

    fn float_permute<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self, D>,
        axes: [usize; D],
    ) -> burn_tensor::ops::FloatTensor<Self, D> {
        let array = tensor.array.permuted_axes(axes.into_dimension());
        NdArrayTensor { array }
    }

    fn float_flip<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self, D>,
        axes: &[usize],
    ) -> burn_tensor::ops::FloatTensor<Self, D> {
        NdArrayOps::flip(tensor, axes)
    }

    fn float_sign<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        NdArrayMathOps::sign_op(tensor)
    }

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> burn_tensor::ops::FloatTensor<Self, D2> {
        NdArrayOps::expand(tensor, shape)
    }
}
