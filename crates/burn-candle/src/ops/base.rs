use std::marker::PhantomData;

use burn_tensor::{Element, Shape, TensorData, TensorMetadata, backend::Backend};
use candle_core::WithDType;
use half::{bf16, f16};

use crate::{
    Candle, CandleDevice, CandleTensor,
    element::{CandleElement, FloatCandleElement, IntCandleElement},
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
    zeros(shape, device, dtype)
}

pub fn zeros(shape: Shape, device: &CandleDevice, dtype: candle_core::DType) -> CandleTensor {
    CandleTensor::new(
        candle_core::Tensor::zeros(shape.dims, dtype, &(device.clone()).into()).unwrap(),
    )
}

pub fn ones(shape: Shape, device: &CandleDevice, dtype: candle_core::DType) -> CandleTensor {
    CandleTensor::new(
        candle_core::Tensor::ones(shape.dims, dtype, &(device.clone()).into()).unwrap(),
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

pub fn slice_with_steps(tensor: CandleTensor, slices: &[burn_tensor::Slice]) -> CandleTensor {
    let mut result_tensor = tensor.tensor;

    for (dim, slice) in slices.iter().enumerate() {
        if slice.step == 1 {
            // Use narrow for step=1 (more efficient)
            // Convert slice to range using tensor shape
            let dim_size = result_tensor.dim(dim).unwrap();
            let range = slice.to_range(dim_size);
            let start = range.start;
            let length = range.end - range.start;
            result_tensor = result_tensor.narrow(dim, start, length).unwrap();
        } else {
            // Use index_select for step != 1
            let dim_size = result_tensor.dim(dim).unwrap();
            let range = slice.to_range(dim_size);
            let start = range.start;
            let end = range.end;
            let step = slice.step;

            // Generate indices based on step direction
            let indices_vec = if step > 0 {
                // Forward stepping
                let step_usize = step as usize;
                (start..end).step_by(step_usize).collect::<Vec<_>>()
            } else {
                // Backward stepping (negative step)
                let step_usize = step.unsigned_abs();
                // Start from end-1 and go backwards
                let mut indices = Vec::new();
                let mut idx = end - 1;
                while idx >= start && idx < end {
                    // Check for underflow
                    indices.push(idx);
                    if idx >= step_usize {
                        idx -= step_usize;
                    } else {
                        break;
                    }
                }
                indices
            };

            // Convert indices to tensor and use index_select
            let indices_len = indices_vec.len();
            let device = result_tensor.device();
            let indices = candle_core::Tensor::from_vec(
                indices_vec.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                indices_len,
                device,
            )
            .unwrap();

            result_tensor = result_tensor.index_select(&indices, dim).unwrap();
        }
    }

    CandleTensor::new(result_tensor)
}

pub fn slice_assign(
    tensor: CandleTensor,
    slices: &[burn_tensor::Slice],
    value: CandleTensor,
) -> CandleTensor {
    // Check if all slices have step=1 (candle's native slice_assign requirement)
    let all_unit_steps = slices.iter().all(|s| s.step == 1);

    if all_unit_steps {
        // Convert Slice to Range for candle's native slice_assign
        let ranges: Vec<std::ops::Range<usize>> = slices
            .iter()
            .enumerate()
            .map(|(dim, slice)| {
                let dim_size = tensor.tensor.dim(dim).unwrap_or(usize::MAX);
                slice.to_range(dim_size)
            })
            .collect();

        CandleTensor::new(tensor.tensor.slice_assign(&ranges, &value.tensor).unwrap())
    } else {
        // Candle doesn't support slice_assign with steps != 1
        // We need to implement it manually or panic
        panic!(
            "Candle backend does not support slice_assign with step != 1 yet. \
                See https://github.com/huggingface/candle/issues/3095"
        );
    }
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
    let mut mask = mask.tensor;

    if shape != *tensor.shape() {
        tensor = tensor.broadcast_as(shape.clone()).unwrap();
    }
    if shape != *mask.shape() {
        mask = mask.broadcast_as(shape).unwrap();
    }

    CandleTensor::new(mask.where_cond(&value.tensor, &tensor).unwrap())
}
