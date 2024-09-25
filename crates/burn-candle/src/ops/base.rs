use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Shape, TensorData};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleDevice, CandleTensor,
};

use super::tensor;

pub fn cat<E: CandleElement>(tensors: Vec<CandleTensor<E>>, dim: usize) -> CandleTensor<E> {
    let tensors: Vec<candle_core::Tensor> = tensors.into_iter().map(|t| t.tensor).collect();
    CandleTensor::new(candle_core::Tensor::cat(&tensors, dim).unwrap())
}

pub fn from_data<E: CandleElement>(data: TensorData, device: &CandleDevice) -> CandleTensor<E> {
    CandleTensor::from_data(data, device.clone())
}
pub fn into_data<E: CandleElement>(tensor: CandleTensor<E>) -> TensorData {
    TensorData::new(
        tensor.tensor.flatten_all().unwrap().to_vec1::<E>().unwrap(),
        tensor.shape(),
    )
}

pub fn to_device<E: CandleElement>(
    tensor: CandleTensor<E>,
    device: &CandleDevice,
) -> CandleTensor<E> {
    CandleTensor::new(tensor.tensor.to_device(&(device.clone()).into()).unwrap())
}

pub fn empty<E: CandleElement>(shape: Shape, device: &CandleDevice) -> CandleTensor<E> {
    CandleTensor::new(
        candle_core::Tensor::zeros(shape.dims, E::DTYPE, &(device.clone()).into()).unwrap(),
    )
}

pub fn swap_dims<E: CandleElement>(
    mut tensor: CandleTensor<E>,
    dim1: usize,
    dim2: usize,
) -> CandleTensor<E> {
    CandleTensor::new(tensor.tensor.transpose(dim1, dim2).unwrap())
}

pub fn permute<E: CandleElement>(tensor: CandleTensor<E>, axes: &[usize]) -> CandleTensor<E> {
    CandleTensor::new(tensor.tensor.permute(axes).unwrap())
}

pub fn flip<E: CandleElement>(tensor: CandleTensor<E>, axes: &[usize]) -> CandleTensor<E> {
    // FIXME: Replace with an appropriate method when Candle provides one.
    let mut tensor = tensor.tensor;
    for &axis in axes {
        let indexes = candle_core::Tensor::arange_step(
            tensor.dim(axis).unwrap() as i64 - 1,
            -1,
            -1,
            tensor.device(),
        )
        .unwrap();
        tensor = tensor.index_select(&indexes, axis).unwrap();
    }

    CandleTensor::new(tensor)
}

pub fn reshape<E: CandleElement>(tensor: CandleTensor<E>, shape: Shape) -> CandleTensor<E> {
    CandleTensor::new(tensor.tensor.reshape(shape.dims).unwrap())
}

pub fn device<E: CandleElement>(tensor: &CandleTensor<E>) -> CandleDevice {
    tensor.tensor.device().clone().into()
}

pub fn shape<E: CandleElement>(tensor: &CandleTensor<E>) -> Shape {
    tensor.shape()
}

pub fn slice<E: CandleElement>(
    tensor: CandleTensor<E>,
    ranges: &[std::ops::Range<usize>],
) -> CandleTensor<E> {
    let mut narrow_tensor = tensor.tensor;
    for (i, range) in ranges.iter().enumerate().take(ranges.len()) {
        narrow_tensor = narrow_tensor
            .narrow(i, range.start, range.end - range.start)
            .unwrap()
    }
    CandleTensor::new(narrow_tensor)
}

pub fn slice_assign<E: CandleElement>(
    tensor: CandleTensor<E>,
    ranges: &[std::ops::Range<usize>],
    value: CandleTensor<E>,
) -> CandleTensor<E> {
    CandleTensor::new(tensor.tensor.slice_assign(ranges, &value.tensor).unwrap())
}

pub fn narrow<E: CandleElement>(
    tensor: CandleTensor<E>,
    dim: usize,
    start: usize,
    length: usize,
) -> CandleTensor<E> {
    let tensor = tensor.tensor.narrow(dim, start, length);
    match tensor {
        Ok(tensor) => CandleTensor::new(tensor),
        Err(e) => panic!("error narrow from Candle"),
    }
}

pub fn chunk<E: CandleElement>(
    tensor: CandleTensor<E>,
    chunks: usize,
    dim: usize,
) -> Vec<CandleTensor<E>> {
    let tensors = tensor.tensor.chunk(chunks, dim);
    match tensors {
        Ok(tensors) => tensors
            .into_iter()
            .map(|tensor| CandleTensor::new(tensor))
            .collect(),
        Err(e) => panic!("error chunk from Candle"),
    }
}

pub fn expand<E: CandleElement>(tensor: CandleTensor<E>, shape: Shape) -> CandleTensor<E> {
    CandleTensor::new(tensor.tensor.broadcast_as(shape.dims).unwrap())
}

pub fn sign<E: CandleElement>(tensor: CandleTensor<E>) -> CandleTensor<E> {
    CandleTensor::new(tensor.tensor.sign().unwrap())
}
