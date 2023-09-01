// Language
use alloc::vec;
use alloc::vec::Vec;
use burn_tensor::ops::IntTensorOps;
use burn_tensor::ElementConversion;
use core::ops::Range;

// Current crate
use crate::element::ExpElement;
use crate::element::FloatNdArrayElement;
use crate::NdArrayDevice;
use crate::{tensor::NdArrayTensor, NdArrayBackend};

// Workspace crates
use burn_tensor::{backend::Backend, Data, Shape};

use super::{NdArrayMathOps, NdArrayOps};

impl<E: FloatNdArrayElement> IntTensorOps<NdArrayBackend<E>> for NdArrayBackend<E> {
    fn int_from_data<const D: usize>(
        data: Data<i64, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<i64, D> {
        NdArrayTensor::from_data(data)
    }

    fn int_shape<const D: usize>(tensor: &NdArrayTensor<i64, D>) -> Shape<D> {
        tensor.shape()
    }

    fn int_to_data<const D: usize>(tensor: &NdArrayTensor<i64, D>) -> Data<i64, D> {
        let values = tensor.array.iter().map(Clone::clone).collect();
        Data::new(values, tensor.shape())
    }

    fn int_into_data<const D: usize>(tensor: NdArrayTensor<i64, D>) -> Data<i64, D> {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        Data::new(values, shape)
    }

    fn int_to_device<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<i64, D> {
        tensor
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<i64, D1>,
        shape: Shape<D2>,
    ) -> NdArrayTensor<i64, D2> {
        NdArrayOps::reshape(tensor, shape)
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<i64, D1>,
        ranges: [Range<usize>; D2],
    ) -> NdArrayTensor<i64, D1> {
        NdArrayOps::slice(tensor, ranges)
    }

    fn int_device<const D: usize>(
        _tensor: &NdArrayTensor<i64, D>,
    ) -> <NdArrayBackend<E> as Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn int_empty<const D: usize>(
        shape: Shape<D>,
        _device: &<NdArrayBackend<E> as Backend>::Device,
    ) -> NdArrayTensor<i64, D> {
        let values = vec![0; shape.num_elements()];
        NdArrayTensor::from_data(Data::new(values, shape))
    }

    fn int_mask_where<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        mask: NdArrayTensor<bool, D>,
        source: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::mask_where(tensor, mask, source)
    }

    fn int_mask_fill<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        mask: NdArrayTensor<bool, D>,
        value: i64,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::mask_fill(tensor, mask, value)
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<i64, D1>,
        ranges: [Range<usize>; D2],
        value: NdArrayTensor<i64, D1>,
    ) -> NdArrayTensor<i64, D1> {
        NdArrayOps::slice_assign(tensor, ranges, value)
    }

    fn int_cat<const D: usize>(
        tensors: Vec<NdArrayTensor<i64, D>>,
        dim: usize,
    ) -> NdArrayTensor<i64, D> {
        NdArrayOps::cat(tensors, dim)
    }

    fn int_equal<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = Self::int_sub(lhs, rhs);

        Self::int_equal_elem(tensor, 0)
    }

    fn int_equal_elem<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: i64,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a == rhs).into_shared();
        NdArrayTensor { array }
    }

    fn int_greater<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_greater_elem(tensor, 0)
    }

    fn int_greater_elem<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: i64,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a > rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_greater_equal<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_greater_equal_elem(tensor, 0)
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: i64,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a >= rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_lower<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_lower_elem(tensor, 0)
    }

    fn int_lower_elem<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: i64,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a < rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_lower_equal<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = Self::int_sub(lhs, rhs);
        Self::int_lower_equal_elem(tensor, 0)
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: i64,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a <= rhs).into_shared();
        NdArrayTensor::new(array)
    }

    fn int_add<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::add(lhs, rhs)
    }

    fn int_add_scalar<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: i64,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::add_scalar(lhs, rhs)
    }

    fn int_sub<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::sub(lhs, rhs)
    }

    fn int_sub_scalar<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: i64,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::sub_scalar(lhs, rhs)
    }

    fn int_mul<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::mul(lhs, rhs)
    }

    fn int_mul_scalar<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: i64,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::mul_scalar(lhs, rhs)
    }

    fn int_div<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::div(lhs, rhs)
    }

    fn int_div_scalar<const D: usize>(
        lhs: NdArrayTensor<i64, D>,
        rhs: i64,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::div_scalar(lhs, rhs)
    }

    fn int_neg<const D: usize>(tensor: NdArrayTensor<i64, D>) -> NdArrayTensor<i64, D> {
        Self::int_mul_scalar(tensor, -1)
    }

    fn int_zeros<const D: usize>(
        shape: Shape<D>,
        device: &<NdArrayBackend<E> as Backend>::Device,
    ) -> NdArrayTensor<i64, D> {
        Self::int_from_data(Data::zeros(shape), device)
    }

    fn int_ones<const D: usize>(
        shape: Shape<D>,
        device: &<NdArrayBackend<E> as Backend>::Device,
    ) -> NdArrayTensor<i64, D> {
        Self::int_from_data(Data::ones(shape), device)
    }

    fn int_full<const D: usize>(
        shape: Shape<D>,
        fill_value: i64,
        device: &<NdArrayBackend<E> as Backend>::Device,
    ) -> NdArrayTensor<i64, D> {
        Self::int_from_data(Data::full(shape, fill_value), device)
    }

    fn int_sum<const D: usize>(tensor: NdArrayTensor<i64, D>) -> NdArrayTensor<i64, 1> {
        NdArrayMathOps::sum(tensor)
    }

    fn int_sum_dim<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        dim: usize,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::sum_dim(tensor, dim)
    }

    fn int_mean<const D: usize>(tensor: NdArrayTensor<i64, D>) -> NdArrayTensor<i64, 1> {
        NdArrayMathOps::mean(tensor)
    }

    fn int_mean_dim<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        dim: usize,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::mean_dim(tensor, dim)
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: NdArrayTensor<i64, D>,
        indices: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::gather(dim, tensor, indices)
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: NdArrayTensor<i64, D>,
        indices: NdArrayTensor<i64, D>,
        value: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::scatter(dim, tensor, indices, value)
    }

    fn int_select<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        dim: usize,
        indices: NdArrayTensor<i64, 1>,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::select(tensor, dim, indices)
    }

    fn int_select_assign<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        dim: usize,
        indices: NdArrayTensor<i64, 1>,
        value: NdArrayTensor<i64, D>,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::select_assign(tensor, dim, indices, value)
    }
    fn int_argmax<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        dim: usize,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::argmax(tensor, dim)
    }

    fn int_argmin<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        dim: usize,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::argmin(tensor, dim)
    }

    fn int_clamp_min<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        min: i64,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::clamp_min(tensor, min)
    }

    fn int_clamp_max<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        max: i64,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::clamp_max(tensor, max)
    }

    fn int_clamp<const D: usize>(
        tensor: NdArrayTensor<i64, D>,
        min: i64,
        max: i64,
    ) -> NdArrayTensor<i64, D> {
        NdArrayMathOps::clamp(tensor, min, max)
    }

    fn int_abs<const D: usize>(tensor: NdArrayTensor<i64, D>) -> NdArrayTensor<i64, D> {
        let array = tensor.array.mapv_into(|a| a.int_abs_elem()).into_shared();

        NdArrayTensor::new(array)
    }

    fn int_into_float<const D: usize>(
        tensor: <NdArrayBackend<E> as Backend>::IntTensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = tensor.array.mapv(|a| a.elem()).into_shared();
        NdArrayTensor { array }
    }
}
