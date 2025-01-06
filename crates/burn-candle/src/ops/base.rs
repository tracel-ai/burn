use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Element, Shape, TensorData, TensorMetadata};
use candle_core::WithDType;
use half::{bf16, f16};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleDevice, CandleTensor,
};

use super::tensor;

pub fn cat(tensors: Vec<CandleTensor>, dim: usize) -> CandleTensor {
    let tensors: Vec<candle_core::Tensor> = tensors.into_iter().map(|t| t.tensor).collect();
    CandleTensor::new(candle_core::Tensor::cat(&tensors, dim).unwrap())
}

pub fn from_data<E: CandleElement>(data: TensorData, device: &CandleDevice) -> CandleTensor {
    CandleTensor::from_data::<E>(data, device.clone())
}
pub fn into_data(tensor: CandleTensor) -> TensorData {
    fn tensor_data_from_dtype<T: WithDType + Element>(tensor: &CandleTensor) -> TensorData {
        TensorData::new(
            tensor.tensor.flatten_all().unwrap().to_vec1::<T>().unwrap(),
            tensor.shape(),
        )
    }

    match tensor.tensor.dtype() {
        candle_core::DType::BF16 => tensor_data_from_dtype::<bf16>(&tensor),
        candle_core::DType::F16 => tensor_data_from_dtype::<f16>(&tensor),
        candle_core::DType::F32 => tensor_data_from_dtype::<f32>(&tensor),
        candle_core::DType::F64 => tensor_data_from_dtype::<f64>(&tensor),
        candle_core::DType::U8 => tensor_data_from_dtype::<u8>(&tensor),
        candle_core::DType::U32 => tensor_data_from_dtype::<u32>(&tensor),
        candle_core::DType::I64 => tensor_data_from_dtype::<i64>(&tensor),
    }
}

pub fn to_device(tensor: CandleTensor, device: &CandleDevice) -> CandleTensor {
    CandleTensor::new(tensor.tensor.to_device(&(device.clone()).into()).unwrap())
}

pub fn empty(shape: Shape, device: &CandleDevice, dtype: candle_core::DType) -> CandleTensor {
    CandleTensor::new(
        candle_core::Tensor::zeros(shape.dims, dtype, &(device.clone()).into()).unwrap(),
    )
}

pub fn swap_dims(mut tensor: CandleTensor, dim1: usize, dim2: usize) -> CandleTensor {
    CandleTensor::new(tensor.tensor.transpose(dim1, dim2).unwrap())
}

pub fn permute(tensor: CandleTensor, axes: &[usize]) -> CandleTensor {
    CandleTensor::new(tensor.tensor.permute(axes).unwrap())
}

pub fn flip(tensor: CandleTensor, axes: &[usize]) -> CandleTensor {
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

pub fn reshape(tensor: CandleTensor, shape: Shape) -> CandleTensor {
    CandleTensor::new(tensor.tensor.reshape(shape.dims).unwrap())
}

pub fn device(tensor: &CandleTensor) -> CandleDevice {
    tensor.tensor.device().clone().into()
}

pub fn shape(tensor: &CandleTensor) -> Shape {
    tensor.shape()
}

pub fn slice(tensor: CandleTensor, ranges: &[std::ops::Range<usize>]) -> CandleTensor {
    let mut narrow_tensor = tensor.tensor;
    for (i, range) in ranges.iter().enumerate().take(ranges.len()) {
        narrow_tensor = narrow_tensor
            .narrow(i, range.start, range.end - range.start)
            .unwrap()
    }
    CandleTensor::new(narrow_tensor)
}

pub fn slice_assign(
    tensor: CandleTensor,
    ranges: &[std::ops::Range<usize>],
    value: CandleTensor,
) -> CandleTensor {
    CandleTensor::new(tensor.tensor.slice_assign(ranges, &value.tensor).unwrap())
}

pub fn narrow(tensor: CandleTensor, dim: usize, start: usize, length: usize) -> CandleTensor {
    let tensor = tensor.tensor.narrow(dim, start, length);
    match tensor {
        Ok(tensor) => CandleTensor::new(tensor),
        Err(e) => panic!("error narrow from Candle"),
    }
}

pub fn chunk(tensor: CandleTensor, chunks: usize, dim: usize) -> Vec<CandleTensor> {
    let tensors = tensor.tensor.chunk(chunks, dim);
    match tensors {
        Ok(tensors) => tensors.into_iter().map(CandleTensor::new).collect(),
        Err(e) => panic!("error chunk from Candle"),
    }
}

pub fn expand(tensor: CandleTensor, shape: Shape) -> CandleTensor {
    CandleTensor::new(tensor.tensor.broadcast_as(shape.dims).unwrap())
}

pub fn sign(tensor: CandleTensor) -> CandleTensor {
    CandleTensor::new(tensor.tensor.sign().unwrap())
}

pub fn mask_where_broadcasted(
    tensor: CandleTensor,
    mask: CandleTensor,
    value: CandleTensor,
) -> CandleTensor {
    let shape = tensor
        .tensor
        .shape()
        .broadcast_shape_binary_op(mask.tensor.shape(), "where_cond")
        .unwrap();

    let mut tensor = tensor.tensor;
    if shape != *tensor.shape() {
        tensor = tensor.broadcast_as(shape).unwrap();
    }

    CandleTensor::new(mask.tensor.where_cond(&value.tensor, &tensor).unwrap())
}

// Taken from: https://github.com/mokeyish/candle-ext/blob/main/src/cumsum.rs
fn cumsum_ext<D: candle_core::shape::Dim>(
    input: &candle_core::Tensor,
    dim: D,
) -> candle_core::Result<candle_core::Tensor> {
    let dim = dim.to_index(input.shape(), "cumsum")?;
    let dim_size = input.dim(dim)?;

    let mut tensors = Vec::with_capacity(dim_size);

    let mut a = input.clone();
    for i in 0..dim_size {
        if i > 0 {
            a = a.narrow(dim, 1, dim_size - i)?;
            let b = input.narrow(dim, 0, dim_size - i)?;
            a = (a + b)?;
        }
        tensors.push(a.narrow(dim, 0, 1)?);
    }
    let cumsum = candle_core::Tensor::cat(&tensors, dim)?;
    Ok(cumsum)
}

/// Cumulative sum (used for int tensors since the default candle implementation uses matmul).
pub fn cumsum(tensor: CandleTensor, dim: usize) -> CandleTensor {
    CandleTensor::new(cumsum_ext(&tensor.tensor, dim).unwrap())
}
