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
    let mut mask = mask.tensor;

    if shape != *tensor.shape() {
        tensor = tensor.broadcast_as(shape.clone()).unwrap();
    }
    if shape != *mask.shape() {
        mask = mask.broadcast_as(shape).unwrap();
    }

    CandleTensor::new(mask.where_cond(&value.tensor, &tensor).unwrap())
}

pub fn cross(lhs: CandleTensor, rhs: CandleTensor, dim: usize) -> CandleTensor {
    let shape_lhs = lhs.shape();
    let shape_rhs = rhs.shape();
    let ndims = shape_lhs.num_dims();

    // Broadcast the shapes except along dim
    let mut broadcast_shape = vec![0; ndims];
    for (i, item) in broadcast_shape.iter_mut().enumerate().take(ndims) {
        if i == dim {
            *item = shape_lhs.dims[i];
        } else {
            let l = shape_lhs.dims[i];
            let r = shape_rhs.dims[i];
            if l == r {
                *item = l;
            } else if l == 1 {
                *item = r;
            } else if r == 1 {
                *item = l;
            } else {
                panic!("Tensors are not broadcastable along dimension {}", i);
            }
        }
    }

    // Broadcast lhs and rhs
    let lhs_broadcast = if shape_lhs == Shape::from(broadcast_shape.clone()) {
        lhs
    } else {
        expand(lhs, Shape::from(broadcast_shape.clone()))
    };
    let rhs_broadcast = if shape_rhs == Shape::from(broadcast_shape.clone()) {
        rhs
    } else {
        expand(rhs, Shape::from(broadcast_shape.clone()))
    };

    // Now, move dim to the last dimension
    let mut perm = (0..ndims).collect::<Vec<_>>();
    perm.remove(dim);
    perm.push(dim);

    let lhs_permuted = permute(lhs_broadcast, &perm);
    let rhs_permuted = permute(rhs_broadcast, &perm);

    // Reshape to (*, 3)
    let total_elements = lhs_permuted.shape().num_elements();
    let batch_size = total_elements / 3;
    let lhs_reshaped = reshape(lhs_permuted, Shape::new([batch_size, 3]));
    let rhs_reshaped = reshape(rhs_permuted, Shape::new([batch_size, 3]));

    // Extract components using narrow and squeeze
    let lhs_0 = CandleTensor::new(
        lhs_reshaped
            .tensor
            .narrow(1, 0, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let lhs_1 = CandleTensor::new(
        lhs_reshaped
            .tensor
            .narrow(1, 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let lhs_2 = CandleTensor::new(
        lhs_reshaped
            .tensor
            .narrow(1, 2, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let rhs_0 = CandleTensor::new(
        rhs_reshaped
            .tensor
            .narrow(1, 0, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let rhs_1 = CandleTensor::new(
        rhs_reshaped
            .tensor
            .narrow(1, 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let rhs_2 = CandleTensor::new(
        rhs_reshaped
            .tensor
            .narrow(1, 2, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );

    // Compute cross product components
    let result_0 = CandleTensor::new(
        lhs_1
            .tensor
            .mul(&rhs_2.tensor)
            .unwrap()
            .sub(&lhs_2.tensor.mul(&rhs_1.tensor).unwrap())
            .unwrap(),
    );
    let result_1 = CandleTensor::new(
        lhs_2
            .tensor
            .mul(&rhs_0.tensor)
            .unwrap()
            .sub(&lhs_0.tensor.mul(&rhs_2.tensor).unwrap())
            .unwrap(),
    );
    let result_2 = CandleTensor::new(
        lhs_0
            .tensor
            .mul(&rhs_1.tensor)
            .unwrap()
            .sub(&lhs_1.tensor.mul(&rhs_0.tensor).unwrap())
            .unwrap(),
    );

    // Stack the components
    let result_0_unsqueezed = CandleTensor::new(result_0.tensor.unsqueeze(1).unwrap());
    let result_1_unsqueezed = CandleTensor::new(result_1.tensor.unsqueeze(1).unwrap());
    let result_2_unsqueezed = CandleTensor::new(result_2.tensor.unsqueeze(1).unwrap());
    let result = cat(
        vec![
            result_0_unsqueezed,
            result_1_unsqueezed,
            result_2_unsqueezed,
        ],
        1,
    );

    // Reshape back to the broadcast shape with dim at the end
    let mut result_shape = broadcast_shape;
    result_shape.remove(dim);
    result_shape.push(3);
    let result_reshaped = reshape(result, Shape::from(result_shape));

    // Permute back
    let mut inv_perm = vec![0; ndims];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }
    permute(result_reshaped, &inv_perm)
}
