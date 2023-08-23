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
        super::base::shape(tensor)
    }

    fn to_data<const D: usize>(tensor: &CandleTensor<F, D>) -> Data<F, D> {
        super::base::to_data(tensor)
    }

    fn device<const D: usize>(tensor: &CandleTensor<F, D>) -> Device<Self> {
        super::base::device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: CandleTensor<F, D>,
        device: &Device<Self>,
    ) -> CandleTensor<F, D> {
        super::base::to_device(tensor, device)
    }

    fn into_int<const D: usize>(tensor: CandleTensor<F, D>) -> IntTensor<Self, D> {
        CandleTensor::new(tensor.tensor.to_dtype(I::DTYPE).unwrap())
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        super::base::empty(shape, device)
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
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        super::base::reshape(tensor, shape)
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
        super::base::slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        super::base::slice_assign(tensor, ranges, value)
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
        let sum = tensor.tensor.sum_all().unwrap().to_scalar::<F>().unwrap();
        CandleTensor::from_data(Data::new([sum].into(), [1].into()), Self::device(&tensor))
    }

    fn sum_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.sum_keepdim(dim).unwrap())
    }

    fn mean_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.mean_keepdim(dim).unwrap())
    }

    fn to_full_precision<const D: usize>(
        tensor: &FloatTensor<Self, D>,
    ) -> FloatTensor<FullPrecisionBackend<Self>, D> {
        CandleTensor::new(tensor.tensor.to_dtype(candle_core::DType::F32).unwrap())
    }

    fn from_full_precision<const D: usize>(
        tensor: FloatTensor<FullPrecisionBackend<Self>, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.to_dtype(F::DTYPE).unwrap())
    }

    fn exp<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.exp().unwrap())
    }

    fn log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.log().unwrap())
    }

    fn log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new((tensor.tensor + 1.).unwrap().log().unwrap())
    }

    fn powf<const D: usize>(tensor: FloatTensor<Self, D>, value: f32) -> FloatTensor<Self, D> {
        panic!("powf not supported by Candle")
    }

    fn sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.sqrt().unwrap())
    }

    fn abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.abs().unwrap())
    }

    fn cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.cos().unwrap())
    }

    fn sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.sin().unwrap())
    }

    fn tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        let e_x = tensor.tensor.exp().unwrap();
        let e_minus_x = tensor.tensor.neg().unwrap().exp().unwrap();
        CandleTensor::new(((e_x.clone() - e_minus_x.clone()).unwrap() / (e_x + e_minus_x)).unwrap())
    }

    fn erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        // TODO submit an issue at Candle
        panic!("erf not supported by Candle")
    }

    fn cat<const D: usize>(tensors: Vec<FloatTensor<Self, D>>, dim: usize) -> FloatTensor<Self, D> {
        super::base::cat(tensors, dim)
    }

    fn argmax<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .argmax_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn argmin<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .argmin_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }
}
