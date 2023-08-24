use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Data, Shape};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    CandleBackend, CandleDevice, CandleTensor,
};

use super::tensor;

pub type FloatElem<B> = <B as Backend>::FloatElem;
pub type Device<B> = <B as Backend>::Device;

pub type FloatTensor<B, const D: usize> = <B as Backend>::TensorPrimitive<D>;

pub type FullPrecisionBackend<B> = <B as Backend>::FullPrecisionBackend;

pub type IntElem<B> = <B as Backend>::IntElem;
pub type IntTensor<B, const D: usize> = <B as Backend>::IntTensorPrimitive<D>;

pub type BoolTensor<B, const D: usize> = <B as Backend>::BoolTensorPrimitive<D>;

pub fn cat<E: CandleElement, const D: usize>(
    tensors: Vec<CandleTensor<E, D>>,
    dim: usize,
) -> CandleTensor<E, D> {
    let tensors: Vec<candle_core::Tensor> = tensors.into_iter().map(|t| t.tensor).collect();
    CandleTensor::new(candle_core::Tensor::cat(&tensors, dim).unwrap())
}

pub fn from_data<E: CandleElement, const D: usize>(
    data: Data<E, D>,
    device: &CandleDevice,
) -> CandleTensor<E, D> {
    CandleTensor::from_data(data, *device)
}

pub fn to_data<E: CandleElement, const D: usize>(tensor: &CandleTensor<E, D>) -> Data<E, D> {
    Data::new(
        tensor.tensor.flatten_all().unwrap().to_vec1().unwrap(),
        tensor.shape(),
    )
}

pub fn to_device<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    device: &CandleDevice,
) -> CandleTensor<E, D> {
    CandleTensor::new(tensor.tensor.to_device(&(*device).into()).unwrap())
}

pub fn empty<E: CandleElement, const D: usize>(
    shape: Shape<D>,
    device: &CandleDevice,
) -> CandleTensor<E, D> {
    CandleTensor::new(candle_core::Tensor::zeros(&shape.dims, E::DTYPE, &(*device).into()).unwrap())
}

pub fn swap_dims<E: CandleElement, const D: usize>(
    mut tensor: CandleTensor<E, D>,
    dim1: usize,
    dim2: usize,
) -> CandleTensor<E, D> {
    CandleTensor::new(tensor.tensor.transpose(dim1, dim2).unwrap())
}

pub fn reshape<E: CandleElement, const D1: usize, const D2: usize>(
    tensor: CandleTensor<E, D1>,
    shape: Shape<D2>,
) -> CandleTensor<E, D2> {
    CandleTensor::new(tensor.tensor.reshape(&shape.dims).unwrap())
}

pub fn device<E: CandleElement, const D: usize>(tensor: &CandleTensor<E, D>) -> CandleDevice {
    tensor.tensor.device().clone().into()
}

pub fn shape<E: CandleElement, const D: usize>(tensor: &CandleTensor<E, D>) -> Shape<D> {
    tensor.shape()
}

pub fn slice<E: CandleElement, const D1: usize, const D2: usize>(
    tensor: CandleTensor<E, D1>,
    ranges: [std::ops::Range<usize>; D2],
) -> CandleTensor<E, D1> {
    let mut narrow_tensor = tensor.tensor;
    for (i, range) in ranges.iter().enumerate().take(D2) {
        narrow_tensor = narrow_tensor
            .narrow(i, range.start, range.end - range.start)
            .unwrap()
    }
    CandleTensor::new(narrow_tensor)
}

pub fn slice_assign<E: CandleElement, const D1: usize, const D2: usize>(
    tensor: CandleTensor<E, D1>,
    ranges: [std::ops::Range<usize>; D2],
    value: CandleTensor<E, D1>,
) -> CandleTensor<E, D1> {
    // TODO: not trivial, because no view_ like in torch
    todo!()
}
