use std::borrow::Borrow;

use burn_tensor::{ops::TensorOps, Data, Distribution, Shape};
use candle_core::{shape, Tensor};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    CandleBackend, CandleTensor,
};

use super::base::{BoolTensor, Device, FloatElem, FloatTensor, FullPrecisionBackend, IntTensor};

impl<F: FloatCandleElement, I: IntCandleElement> TensorOps<CandleBackend<F, I>>
    for CandleBackend<F, I>
{
    fn from_data<const D: usize>(data: Data<F, D>, device: &Device<Self>) -> CandleTensor<F, D> {
        CandleTensor::from_data(data, *device)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<F>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn shape<const D: usize>(tensor: &CandleTensor<F, D>) -> Shape<D> {
        tensor.shape()
    }

    fn to_data<const D: usize>(tensor: &CandleTensor<F, D>) -> Data<F, D> {
        Data::new(
            tensor.tensor.flatten_all().unwrap().to_vec1().unwrap(),
            tensor.shape(),
        )
    }

    fn device<const D: usize>(tensor: &CandleTensor<F, D>) -> Device<Self> {
        tensor.tensor.device().clone().into()
    }

    fn to_device<const D: usize>(
        tensor: CandleTensor<F, D>,
        device: &Device<Self>,
    ) -> CandleTensor<F, D> {
        CandleTensor::new(tensor.tensor.to_device(&(*device).into()).unwrap())
    }

    fn into_int<const D: usize>(tensor: CandleTensor<F, D>) -> IntTensor<Self, D> {
        CandleTensor::new(tensor.tensor.to_dtype(I::DTYPE).unwrap())
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        CandleTensor::new(
            candle_core::Tensor::zeros(&shape.dims, F::DTYPE, &(*device).into()).unwrap(),
        )
    }

    fn add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_add(&rhs.tensor).unwrap())
    }

    fn add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new((lhs.tensor + rhs.elem::<f64>()).unwrap())
    }

    fn sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_sub(&rhs.tensor).unwrap())
    }

    fn sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new((lhs.tensor - rhs.elem::<f64>()).unwrap())
    }

    fn mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_mul(&rhs.tensor).unwrap())
    }

    fn mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new((lhs.tensor * rhs.elem::<f64>()).unwrap())
    }

    fn div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_div(&rhs.tensor).unwrap())
    }

    fn div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new((lhs.tensor / rhs.elem::<f64>()).unwrap())
    }

    fn matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_matmul(&rhs.tensor).unwrap())
    }

    fn swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.transpose(dim1, dim2).unwrap())
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        CandleTensor::new(tensor.tensor.reshape(&shape.dims).unwrap())
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.gather(&indices.tensor, dim).unwrap())
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .scatter_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.index_select(&indices.tensor, dim).unwrap())
    }

    fn select_assign<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .index_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        let mut narrow_tensor = tensor.tensor;
        for (i, range) in ranges.iter().enumerate().take(D2) {
            narrow_tensor = narrow_tensor
                .narrow(i, range.start, range.end - range.start)
                .unwrap()
        }
        CandleTensor::new(narrow_tensor)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        // TODO: not trivial, because no view_ like in torch
        todo!()
    }

    fn mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(
            mask.tensor
                .where_cond(&value.tensor, &tensor.tensor)
                .unwrap(),
        )
    }

    fn mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(
            mask.tensor
                .where_cond(
                    &(candle_core::Tensor::ones_like(&tensor.tensor).unwrap()
                        * value.elem::<f64>())
                    .unwrap(),
                    &tensor.tensor,
                )
                .unwrap(),
        )
    }

    fn equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.eq(&rhs.tensor).unwrap())
    }

    fn equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .eq(
                    &(candle_core::Tensor::ones_like(&lhs.tensor).unwrap() * rhs.elem::<f64>())
                        .unwrap(),
                )
                .unwrap(),
        )
    }

    fn greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.gt(&rhs.tensor).unwrap())
    }

    fn greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .gt(
                    &(candle_core::Tensor::ones_like(&lhs.tensor).unwrap() * rhs.elem::<f64>())
                        .unwrap(),
                )
                .unwrap(),
        )
    }

    fn greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.ge(&rhs.tensor).unwrap())
    }

    fn greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .ge(
                    &(candle_core::Tensor::ones_like(&lhs.tensor).unwrap() * rhs.elem::<f64>())
                        .unwrap(),
                )
                .unwrap(),
        )
    }

    fn lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.lt(&rhs.tensor).unwrap())
    }

    fn lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .lt(
                    &(candle_core::Tensor::ones_like(&lhs.tensor).unwrap() * rhs.elem::<f64>())
                        .unwrap(),
                )
                .unwrap(),
        )
    }

    fn lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.le(&rhs.tensor).unwrap())
    }

    fn lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .le(
                    &(candle_core::Tensor::ones_like(&lhs.tensor).unwrap() * rhs.elem::<f64>())
                        .unwrap(),
                )
                .unwrap(),
        )
    }

    fn sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        todo!()
    }

    fn sum_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        todo!()
    }

    fn mean_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        todo!()
    }

    fn to_full_precision<const D: usize>(
        tensor: &FloatTensor<Self, D>,
    ) -> FloatTensor<FullPrecisionBackend<Self>, D> {
        todo!()
    }

    fn from_full_precision<const D: usize>(
        tensor: FloatTensor<FullPrecisionBackend<Self>, D>,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn exp<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn powf<const D: usize>(tensor: FloatTensor<Self, D>, value: f32) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn cat<const D: usize>(tensors: Vec<FloatTensor<Self, D>>, dim: usize) -> FloatTensor<Self, D> {
        todo!()
    }

    fn argmax<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        todo!()
    }

    fn argmin<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        todo!()
    }
}
