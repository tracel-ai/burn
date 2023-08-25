use burn_tensor::{ops::BoolTensorOps, Data, Shape};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    CandleBackend, CandleTensor,
};

use super::base::{BoolTensor, Device, FloatTensor, IntTensor};

impl<F: FloatCandleElement, I: IntCandleElement> BoolTensorOps<CandleBackend<F, I>>
    for CandleBackend<F, I>
{
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> BoolTensor<Self, D> {
        super::base::empty(shape, device)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<Self, D>) -> Shape<D> {
        super::base::shape(tensor)
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<Self, D>) -> Data<bool, D> {
        let x: Vec<u8> = tensor.tensor.flatten_all().unwrap().to_vec1().unwrap();
        let y = x.iter().map(|b| !matches!(b, 0)).collect();
        Data::new(y, tensor.shape())
    }

    fn bool_from_data<const D: usize>(
        data: Data<bool, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        let data: Data<u8, D> = Data::new(
            data.value
                .into_iter()
                .map(|c| match c {
                    true => 1,
                    false => 0,
                })
                .collect(),
            data.shape,
        );
        super::base::from_data(data, device)
    }

    fn bool_into_int<const D: usize>(tensor: BoolTensor<Self, D>) -> IntTensor<Self, D> {
        CandleTensor::new(tensor.tensor.to_dtype(I::DTYPE).unwrap())
    }

    fn bool_into_float<const D: usize>(tensor: BoolTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.to_dtype(F::DTYPE).unwrap())
    }

    fn bool_device<const D: usize>(tensor: &BoolTensor<Self, D>) -> Device<Self> {
        super::base::device(tensor)
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<Self, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        super::base::to_device(tensor, device)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<Self, D2> {
        super::base::reshape(tensor, shape)
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> BoolTensor<Self, D1> {
        super::base::slice(tensor, ranges)
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: BoolTensor<Self, D1>,
    ) -> BoolTensor<Self, D1> {
        super::base::slice_assign(tensor, ranges, value)
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<BoolTensor<Self, D>>,
        dim: usize,
    ) -> BoolTensor<Self, D> {
        super::base::cat(tensors, dim)
    }

    fn bool_equal<const D: usize>(
        lhs: BoolTensor<Self, D>,
        rhs: BoolTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.eq(&rhs.tensor).unwrap())
    }

    fn bool_equal_elem<const D: usize>(lhs: BoolTensor<Self, D>, rhs: bool) -> BoolTensor<Self, D> {
        let rhs: f64 = match rhs {
            false => 0.,
            true => 1.,
        };
        let x = (candle_core::Tensor::ones_like(&lhs.tensor).unwrap() * rhs).unwrap();
        CandleTensor::new(lhs.tensor.eq(&x).unwrap())
    }
}
