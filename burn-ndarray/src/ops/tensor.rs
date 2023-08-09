// Language
use alloc::vec::Vec;
use core::ops::Range;

// Current crate
use super::{matmul::matmul, NdArrayMathOps, NdArrayOps};
use crate::element::FloatNdArrayElement;
use crate::{tensor::NdArrayTensor, NdArrayBackend};
use crate::{NdArrayDevice, SEED};

// Workspace crates
use burn_common::rand::get_seeded_rng;
use burn_tensor::Distribution;
use burn_tensor::{backend::Backend, ops::TensorOps, Data, ElementConversion, Shape};

// External crates
use libm::{cos, erf, sin, tanh};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

impl<E: FloatNdArrayElement> TensorOps<NdArrayBackend<E>> for NdArrayBackend<E> {
    fn from_data<const D: usize>(data: Data<E, D>, _device: &NdArrayDevice) -> NdArrayTensor<E, D> {
        NdArrayTensor::from_data(data)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<E>,
        device: &NdArrayDevice,
    ) -> NdArrayTensor<E, D> {
        let mut seed = SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            get_seeded_rng()
        };
        let tensor = Self::from_data(Data::random(shape, distribution, &mut rng), device);
        *seed = Some(rng);
        tensor
    }

    fn shape<const D: usize>(tensor: &NdArrayTensor<E, D>) -> Shape<D> {
        tensor.shape()
    }

    fn to_data<const D: usize>(
        tensor: &NdArrayTensor<E, D>,
    ) -> Data<<NdArrayBackend<E> as Backend>::FloatElem, D> {
        let values = tensor.array.iter().map(Clone::clone).collect();
        Data::new(values, tensor.shape())
    }

    fn into_data<const D: usize>(
        tensor: NdArrayTensor<E, D>,
    ) -> Data<<NdArrayBackend<E> as Backend>::FloatElem, D> {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        Data::new(values, shape)
    }

    fn device<const D: usize>(_tensor: &NdArrayTensor<E, D>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn to_device<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<E, D> {
        tensor
    }

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<NdArrayBackend<E> as Backend>::Device,
    ) -> NdArrayTensor<E, D> {
        NdArrayBackend::<E>::zeros(shape, device)
    }

    fn add<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::add(lhs, rhs)
    }

    fn add_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::add_scalar(lhs, rhs)
    }

    fn sub<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::sub(lhs, rhs)
    }

    fn sub_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::sub_scalar(lhs, rhs)
    }

    fn mul<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mul(lhs, rhs)
    }

    fn mul_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mul_scalar(lhs, rhs)
    }

    fn div<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::div(lhs, rhs)
    }

    fn div_scalar<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::div_scalar(lhs, rhs)
    }

    fn matmul<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        matmul(lhs, rhs)
    }

    fn neg<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        Self::mul_scalar(tensor, (-1f32).elem::<E>())
    }

    fn swap_dims<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> NdArrayTensor<E, D> {
        let mut array = tensor.array;
        array.swap_axes(dim1, dim2);

        NdArrayTensor::new(array)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        shape: Shape<D2>,
    ) -> NdArrayTensor<E, D2> {
        NdArrayOps::reshape(tensor, shape)
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: NdArrayTensor<E, D>,
        indices: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::gather(dim, tensor, indices)
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: NdArrayTensor<E, D>,
        indices: NdArrayTensor<i64, D>,
        value: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::scatter(dim, tensor, indices, value)
    }

    fn select<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
        indices: NdArrayTensor<i64, 1>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::select(tensor, dim, indices)
    }

    fn select_assign<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        dim: usize,
        indices: NdArrayTensor<i64, 1>,
        value: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::select_assign(tensor, dim, indices, value)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        ranges: [Range<usize>; D2],
    ) -> NdArrayTensor<E, D1> {
        NdArrayOps::slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<E, D1>,
        ranges: [Range<usize>; D2],
        value: NdArrayTensor<E, D1>,
    ) -> NdArrayTensor<E, D1> {
        NdArrayOps::slice_assign(tensor, ranges, value)
    }

    fn mask_where<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        mask: NdArrayTensor<bool, D>,
        value: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mask_where(tensor, mask, value)
    }

    fn mask_fill<const D: usize>(
        tensor: NdArrayTensor<E, D>,
        mask: NdArrayTensor<bool, D>,
        value: E,
    ) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mask_fill(tensor, mask, value)
    }

    fn equal<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = 0.elem();

        Self::equal_elem(tensor, zero)
    }

    fn equal_elem<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a == rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn greater<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = 0.elem();
        Self::greater_elem(tensor, zero)
    }

    fn greater_elem<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a > rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn greater_equal<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = 0.elem();
        Self::greater_equal_elem(tensor, zero)
    }

    fn greater_equal_elem<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a >= rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn lower<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = 0.elem();
        Self::lower_elem(tensor, zero)
    }

    fn lower_elem<const D: usize>(lhs: NdArrayTensor<E, D>, rhs: E) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a < rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn lower_equal<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = 0.elem();
        Self::lower_equal_elem(tensor, zero)
    }

    fn lower_equal_elem<const D: usize>(
        lhs: NdArrayTensor<E, D>,
        rhs: E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a <= rhs).into_shared();

        NdArrayTensor::new(array)
    }

    fn detach<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        tensor
    }

    fn mean<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, 1> {
        NdArrayMathOps::mean(tensor)
    }

    fn sum<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, 1> {
        NdArrayMathOps::sum(tensor)
    }

    fn mean_dim<const D: usize>(tensor: NdArrayTensor<E, D>, dim: usize) -> NdArrayTensor<E, D> {
        NdArrayMathOps::mean_dim(tensor, dim)
    }

    fn sum_dim<const D: usize>(tensor: NdArrayTensor<E, D>, dim: usize) -> NdArrayTensor<E, D> {
        NdArrayMathOps::sum_dim(tensor, dim)
    }

    fn to_full_precision<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<f32, D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn from_full_precision<const D: usize>(tensor: NdArrayTensor<f32, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn argmax<const D: usize>(tensor: NdArrayTensor<E, D>, dim: usize) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::argmax(tensor, dim)
    }

    fn argmin<const D: usize>(tensor: NdArrayTensor<E, D>, dim: usize) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::argmin(tensor, dim)
    }

    fn exp<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.exp_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn log<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.log_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn log1p<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.log1p_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn powf<const D: usize>(tensor: NdArrayTensor<E, D>, value: f32) -> NdArrayTensor<E, D> {
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

    fn sqrt<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.sqrt_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn abs<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv_into(|a| a.abs_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn cos<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv_into(|a| cos(a.to_f64().unwrap()).elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn sin<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv_into(|a| sin(a.to_f64().unwrap()).elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn tanh<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv_into(|a| tanh(a.to_f64().unwrap()).elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn erf<const D: usize>(tensor: NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv_into(|a| erf(a.to_f64().unwrap()).elem())
            .into_shared();

        NdArrayTensor::new(array)
    }

    fn cat<const D: usize>(tensors: Vec<NdArrayTensor<E, D>>, dim: usize) -> NdArrayTensor<E, D> {
        NdArrayOps::cat(tensors, dim)
    }

    fn clamp_min<const D: usize>(tensor: NdArrayTensor<E, D>, min: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::clamp_min(tensor, min)
    }

    fn clamp_max<const D: usize>(tensor: NdArrayTensor<E, D>, max: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::clamp_max(tensor, max)
    }

    fn clamp<const D: usize>(tensor: NdArrayTensor<E, D>, min: E, max: E) -> NdArrayTensor<E, D> {
        NdArrayMathOps::clamp(tensor, min, max)
    }

    fn into_int<const D: usize>(
        tensor: <NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::IntTensorPrimitive<D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();
        NdArrayTensor { array }
    }
}
