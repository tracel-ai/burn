use burn_tensor::{
    ops::{BoolTensor, BoolTensorOps, FloatTensor, IntTensor},
    Data, Device, Reader, Shape,
};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleTensor,
};

use super::base::{expand, permute};

impl<F: FloatCandleElement, I: IntCandleElement> BoolTensorOps<Self> for Candle<F, I> {
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> BoolTensor<Self, D> {
        super::base::empty(shape, device)
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<Self, D>) -> Shape<D> {
        super::base::shape(tensor)
    }

    fn bool_into_data<const D: usize>(tensor: BoolTensor<Self, D>) -> Reader<Data<bool, D>> {
        let x: Vec<u8> = tensor.tensor.flatten_all().unwrap().to_vec1().unwrap();
        let y = x.iter().map(|b| !matches!(b, 0)).collect();
        let data = Data::new(y, tensor.shape());

        Reader::Concrete(data)
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

    fn bool_not<const D: usize>(tensor: BoolTensor<Self, D>) -> BoolTensor<Self, D> {
        let x = (candle_core::Tensor::zeros_like(&tensor.tensor).unwrap());
        CandleTensor::new(tensor.tensor.eq(&x).unwrap())
    }

    fn bool_swap_dims<const D: usize>(
        tensor: <Candle<F, I> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <Candle<F, I> as burn_tensor::backend::Backend>::BoolTensorPrimitive<D> {
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn bool_narrow<const D: usize>(
        tensor: BoolTensor<Self, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> BoolTensor<Self, D> {
        super::base::narrow(tensor, dim, start, length)
    }

    fn bool_chunk<const D: usize>(
        tensor: BoolTensor<Self, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<BoolTensor<Self, D>> {
        super::base::chunk(tensor, chunks, dim)
    }

    fn bool_permute<const D: usize>(
        tensor: BoolTensor<Self, D>,
        axes: [usize; D],
    ) -> BoolTensor<Self, D> {
        super::base::permute(tensor, axes)
    }

    fn bool_flip<const D: usize>(
        tensor: BoolTensor<Self, D>,
        axes: &[usize],
    ) -> BoolTensor<Self, D> {
        super::base::flip(tensor, axes)
    }

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<Self, D2> {
        expand(tensor, shape)
    }
}
