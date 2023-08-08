// Language
use alloc::vec;
use alloc::vec::Vec;
use burn_tensor::ops::{BoolTensorOps, IntTensorOps};
use burn_tensor::ElementConversion;
use core::ops::Range;

// Current crate
use crate::element::FloatNdArrayElement;
use crate::NdArrayDevice;
use crate::{tensor::NdArrayTensor, NdArrayBackend};

// Workspace crates
use burn_tensor::{backend::Backend, Data, Shape};

use super::NdArrayOps;

impl<E: FloatNdArrayElement> BoolTensorOps<NdArrayBackend<E>> for NdArrayBackend<E> {
    fn bool_from_data<const D: usize>(
        data: Data<bool, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<bool, D> {
        NdArrayTensor::from_data(data)
    }

    fn bool_shape<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Shape<D> {
        tensor.shape()
    }

    fn bool_to_data<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let values = tensor.array.iter().map(Clone::clone).collect();
        Data::new(values, tensor.shape())
    }

    fn bool_into_data<const D: usize>(
        tensor: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        Data::new(values, shape)
    }

    fn bool_to_device<const D: usize>(
        tensor: NdArrayTensor<bool, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<bool, D> {
        tensor
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<bool, D1>,
        shape: Shape<D2>,
    ) -> NdArrayTensor<bool, D2> {
        NdArrayOps::reshape(tensor, shape)
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: NdArrayTensor<bool, D1>,
        ranges: [Range<usize>; D2],
    ) -> NdArrayTensor<bool, D1> {
        NdArrayOps::slice(tensor, ranges)
    }

    fn bool_into_int<const D: usize>(
        tensor: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> NdArrayTensor<i64, D> {
        let data = Self::bool_into_data(tensor);
        NdArrayBackend::<E>::int_from_data(data.convert(), &NdArrayDevice::Cpu)
    }

    fn bool_device<const D: usize>(
        _tensor: &<NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn bool_empty<const D: usize>(
        shape: Shape<D>,
        _device: &<NdArrayBackend<E> as Backend>::Device,
    ) -> <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D> {
        let values = vec![false; shape.num_elements()];
        NdArrayTensor::from_data(Data::new(values, shape))
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D1>,
        ranges: [Range<usize>; D2],
        value: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D1>,
    ) -> <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D1> {
        NdArrayOps::slice_assign(tensor, ranges, value)
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<<NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>>,
        dim: usize,
    ) -> <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D> {
        NdArrayOps::cat(tensors, dim)
    }

    fn bool_equal<const D: usize>(
        lhs: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
        rhs: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D> {
        let mut array = lhs.array;
        array.zip_mut_with(&rhs.array, |a, b| *a = *a && *b);

        NdArrayTensor { array }
    }

    fn bool_equal_elem<const D: usize>(
        lhs: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
        rhs: bool,
    ) -> <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D> {
        let array = lhs.array.mapv(|a| a == rhs).into_shared();
        NdArrayTensor { array }
    }

    fn bool_into_float<const D: usize>(
        tensor: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = tensor.array.mapv(|a| (a as i32).elem()).into_shared();
        NdArrayTensor { array }
    }
}
