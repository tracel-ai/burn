use burn_common::future::DynFut;
use burn_tensor::{
    Device, Shape, TensorData, TensorMetadata,
    ops::{BoolTensor, BoolTensorOps, FloatTensor, IntTensor},
};

use crate::{
    Candle, CandleTensor,
    element::{CandleElement, FloatCandleElement, IntCandleElement},
};

use super::base::{expand, permute};

impl<F: FloatCandleElement, I: IntCandleElement> BoolTensorOps<Self> for Candle<F, I> {
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::base::empty(shape, device, candle_core::DType::U8)
    }

    fn bool_zeros(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::base::zeros(shape, device, candle_core::DType::U8)
    }

    fn bool_ones(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::base::ones(shape, device, candle_core::DType::U8)
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> TensorData {
        let x: Vec<u8> = tensor.tensor.flatten_all().unwrap().to_vec1().unwrap();
        let y = x.iter().map(|b| !matches!(b, 0)).collect();
        TensorData::new(y, tensor.shape())
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        match data.dtype {
            burn_tensor::DType::U8 => super::base::from_data::<u8>(data, device),
            _ => unimplemented!("Unsupported dtype for `bool_from_data`"),
        }
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        CandleTensor::new(tensor.tensor.to_dtype(I::DTYPE).unwrap())
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.to_dtype(F::DTYPE).unwrap())
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self> {
        super::base::device(tensor)
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self> {
        super::base::to_device(tensor, device)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        super::base::reshape(tensor, shape)
    }

    fn bool_slice(tensor: BoolTensor<Self>, ranges: &[std::ops::Range<usize>]) -> BoolTensor<Self> {
        super::base::slice(tensor, ranges)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        ranges: &[std::ops::Range<usize>],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        super::base::slice_assign(tensor, ranges, value)
    }

    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        super::base::cat(tensors, dim)
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.eq(&rhs.tensor).unwrap())
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let x = (candle_core::Tensor::zeros_like(&tensor.tensor).unwrap());
        CandleTensor::new(tensor.tensor.eq(&x).unwrap())
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let x = candle_core::Tensor::ones_like(&lhs.tensor).unwrap();
        CandleTensor::new(lhs.tensor.add(&rhs.tensor).unwrap().gt(&x).unwrap())
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .add(&rhs.tensor)
                .unwrap()
                .clamp(0u32, 1u32)
                .unwrap(),
        )
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        super::base::permute(tensor, axes)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        super::base::flip(tensor, axes)
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        expand(tensor, shape)
    }
}
