// Language
use alloc::vec;
use alloc::vec::Vec;
use burn_tensor::ops::{BoolTensorOps, IntTensorOps};
use burn_tensor::ElementConversion;
use core::ops::Range;
use ndarray::{IntoDimension, Zip};

// Current crate
use crate::element::{FloatNdArrayElement, QuantElement};
use crate::NdArrayDevice;
use crate::{tensor::NdArrayTensor, NdArray};

// Workspace crates
use burn_tensor::{backend::Backend, Shape, TensorData};

use super::NdArrayOps;

impl<E: FloatNdArrayElement, Q: QuantElement> BoolTensorOps<Self> for NdArray<E, Q> {
    fn bool_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor<bool> {
        NdArrayTensor::from_data(data)
    }

    fn bool_shape(tensor: &NdArrayTensor<bool>) -> Shape {
        tensor.shape()
    }

    async fn bool_into_data(tensor: NdArrayTensor<bool>) -> TensorData {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        TensorData::new(values, shape)
    }

    fn bool_to_device(tensor: NdArrayTensor<bool>, _device: &NdArrayDevice) -> NdArrayTensor<bool> {
        tensor
    }

    fn bool_reshape(tensor: NdArrayTensor<bool>, shape: Shape) -> NdArrayTensor<bool> {
        NdArrayOps::reshape(tensor, shape)
    }

    fn bool_slice(tensor: NdArrayTensor<bool>, ranges: &[Range<usize>]) -> NdArrayTensor<bool> {
        NdArrayOps::slice(tensor, ranges)
    }

    fn bool_into_int(tensor: NdArrayTensor<bool>) -> NdArrayTensor<i64> {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        NdArray::<E>::int_from_data(
            TensorData::new(values, shape).convert::<i64>(),
            &NdArrayDevice::Cpu,
        )
    }

    fn bool_device(_tensor: &NdArrayTensor<bool>) -> <NdArray<E> as Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn bool_empty(shape: Shape, _device: &<NdArray<E> as Backend>::Device) -> NdArrayTensor<bool> {
        let values = vec![false; shape.num_elements()];
        NdArrayTensor::from_data(TensorData::new(values, shape))
    }

    fn bool_slice_assign(
        tensor: NdArrayTensor<bool>,
        ranges: &[Range<usize>],
        value: NdArrayTensor<bool>,
    ) -> NdArrayTensor<bool> {
        NdArrayOps::slice_assign(tensor, ranges, value)
    }

    fn bool_cat(tensors: Vec<NdArrayTensor<bool>>, dim: usize) -> NdArrayTensor<bool> {
        NdArrayOps::cat(tensors, dim)
    }

    fn bool_equal(lhs: NdArrayTensor<bool>, rhs: NdArrayTensor<bool>) -> NdArrayTensor<bool> {
        let output = Zip::from(&lhs.array)
            .and(&rhs.array)
            .map_collect(|&lhs_val, &rhs_val| (lhs_val == rhs_val))
            .into_shared();
        NdArrayTensor::new(output)
    }

    fn bool_not(tensor: NdArrayTensor<bool>) -> NdArrayTensor<bool> {
        let array = tensor.array.mapv(|a| !a).into_shared();
        NdArrayTensor { array }
    }

    fn bool_into_float(
        tensor: NdArrayTensor<bool>,
    ) -> <NdArray<E> as Backend>::FloatTensorPrimitive {
        let array = tensor.array.mapv(|a| (a as i32).elem()).into_shared();
        NdArrayTensor { array }
    }

    fn bool_swap_dims(
        tensor: NdArrayTensor<bool>,
        dim1: usize,
        dim2: usize,
    ) -> NdArrayTensor<bool> {
        NdArrayOps::swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(tensor: NdArrayTensor<bool>, axes: &[usize]) -> NdArrayTensor<bool> {
        let array = tensor.array.permuted_axes(axes.into_dimension());
        NdArrayTensor { array }
    }

    fn bool_expand(tensor: NdArrayTensor<bool>, shape: Shape) -> NdArrayTensor<bool> {
        NdArrayOps::expand(tensor, shape)
    }

    fn bool_flip(tensor: NdArrayTensor<bool>, axes: &[usize]) -> NdArrayTensor<bool> {
        NdArrayOps::flip(tensor, axes)
    }
}
