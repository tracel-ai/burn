use burn_tensor::{
    ops::{BoolTensor, BoolTensorOps, FloatTensor, IntTensor},
    Device, Shape, TensorData,
};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleTensor,
};

use super::base::{expand, permute};

impl<F: FloatCandleElement, I: IntCandleElement> BoolTensorOps<Self> for Candle<F, I> {
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::base::empty(shape, device)
    }

    fn bool_shape(tensor: &BoolTensor<Self>) -> Shape {
        super::base::shape(tensor)
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> TensorData {
        let x: Vec<u8> = tensor.tensor.flatten_all().unwrap().to_vec1().unwrap();
        let y = x.iter().map(|b| !matches!(b, 0)).collect();
        TensorData::new(y, tensor.shape())
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        let data: TensorData = TensorData::new(data.iter::<bool>().collect(), data.shape);
        super::base::from_data(data, device)
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

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn bool_narrow(
        tensor: BoolTensor<Self>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> BoolTensor<Self> {
        super::base::narrow(tensor, dim, start, length)
    }

    fn bool_chunk(tensor: BoolTensor<Self>, chunks: usize, dim: usize) -> Vec<BoolTensor<Self>> {
        super::base::chunk(tensor, chunks, dim)
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
