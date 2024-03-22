use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Data, Reader, Shape};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleDevice, CandleTensor,
};

use super::tensor;

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
pub fn into_data<E: CandleElement, const D: usize>(tensor: CandleTensor<E, D>) -> Data<E, D> {
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

pub fn permute<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    axes: [usize; D],
) -> CandleTensor<E, D> {
    CandleTensor::new(tensor.tensor.permute(axes).unwrap())
}

pub fn flip<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    axes: &[usize],
) -> CandleTensor<E, D> {
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
    CandleTensor::new(tensor.tensor.slice_assign(&ranges, &value.tensor).unwrap())
}

pub fn narrow<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    dim: usize,
    start: usize,
    length: usize,
) -> CandleTensor<E, D> {
    let tensor = tensor.tensor.narrow(dim, start, length);
    match tensor {
        Ok(tensor) => CandleTensor::new(tensor),
        Err(e) => panic!("error narrow from Candle"),
    }
}

pub fn chunk<E: CandleElement, const D: usize>(
    tensor: CandleTensor<E, D>,
    chunks: usize,
    dim: usize,
) -> Vec<CandleTensor<E, D>> {
    let tensors = tensor.tensor.chunk(chunks, dim);
    match tensors {
        Ok(tensors) => tensors
            .into_iter()
            .map(|tensor| CandleTensor::new(tensor))
            .collect(),
        Err(e) => panic!("error chunk from Candle"),
    }
}

pub fn expand<E: CandleElement, const D1: usize, const D2: usize>(
    tensor: CandleTensor<E, D1>,
    shape: Shape<D2>,
) -> CandleTensor<E, D2> {
    CandleTensor::new(tensor.tensor.broadcast_as(&shape.dims).unwrap())
}
