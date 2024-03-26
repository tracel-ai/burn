use std::borrow::Borrow;

use burn_tensor::{
    ops::{BoolTensor, FloatElem, FloatTensor, FloatTensorOps, FullPrecisionBackend, IntTensor},
    Data, Device, Distribution, ElementConversion, Reader, Shape,
};
use candle_core::{backend::BackendStorage, shape, Tensor};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleTensor,
};

use super::base::{expand, permute};

impl<F: FloatCandleElement, I: IntCandleElement> FloatTensorOps<Self> for Candle<F, I> {
    fn float_from_data<const D: usize>(
        data: Data<F, D>,
        device: &Device<Self>,
    ) -> CandleTensor<F, D> {
        CandleTensor::from_data(data, *device)
    }

    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let shape = &shape.dims;
        let device = &(*device).into();
        match distribution {
            Distribution::Default => CandleTensor::new(
                candle_core::Tensor::rand(0.elem::<F>(), 1.elem::<F>(), shape, device)
                    .unwrap()
                    .to_dtype(F::DTYPE)
                    .unwrap(),
            ),
            Distribution::Bernoulli(prob) => CandleTensor::new(
                candle_core::Tensor::rand(0.elem::<F>(), 1.elem::<F>(), shape, device)
                    .unwrap()
                    .to_dtype(F::DTYPE)
                    .unwrap()
                    .lt(&super::candle_utils::fill(prob, shape, F::DTYPE, device))
                    .unwrap()
                    .to_dtype(F::DTYPE)
                    .unwrap(),
            ),
            Distribution::Uniform(from, to) => CandleTensor::new(
                candle_core::Tensor::rand(from.elem::<F>(), to.elem::<F>(), shape, device).unwrap(),
            ),
            Distribution::Normal(mean, std) => CandleTensor::new(
                candle_core::Tensor::randn(mean.elem::<F>(), std.elem::<F>(), shape, device)
                    .unwrap(),
            ),
        }
    }

    fn float_shape<const D: usize>(tensor: &CandleTensor<F, D>) -> Shape<D> {
        super::base::shape(tensor)
    }

    fn float_into_data<const D: usize>(tensor: CandleTensor<F, D>) -> Reader<Data<F, D>> {
        Reader::Concrete(super::base::into_data(tensor))
    }

    fn float_device<const D: usize>(tensor: &CandleTensor<F, D>) -> Device<Self> {
        super::base::device(tensor)
    }

    fn float_to_device<const D: usize>(
        tensor: CandleTensor<F, D>,
        device: &Device<Self>,
    ) -> CandleTensor<F, D> {
        super::base::to_device(tensor, device)
    }

    fn float_into_int<const D: usize>(tensor: CandleTensor<F, D>) -> IntTensor<Self, D> {
        CandleTensor::new(tensor.tensor.to_dtype(I::DTYPE).unwrap())
    }

    fn float_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        super::base::empty(shape, device)
    }

    fn float_add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_add(&rhs.tensor).unwrap())
    }

    fn float_add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new((lhs.tensor + rhs.elem::<f64>()).unwrap())
    }

    fn float_sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_sub(&rhs.tensor).unwrap())
    }

    fn float_sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new((lhs.tensor - rhs.elem::<f64>()).unwrap())
    }

    fn float_mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_mul(&rhs.tensor).unwrap())
    }

    fn float_mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new((lhs.tensor * rhs.elem::<f64>()).unwrap())
    }

    fn float_div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_div(&rhs.tensor).unwrap())
    }

    fn float_div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new((lhs.tensor / rhs.elem::<f64>()).unwrap())
    }

    fn float_matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        let lhs_contiguous = if !lhs.tensor.is_contiguous() {
            lhs.tensor.contiguous().unwrap()
        } else {
            lhs.tensor
        };
        let rhs_contiguous = if !rhs.tensor.is_contiguous() {
            rhs.tensor.contiguous().unwrap()
        } else {
            rhs.tensor
        };
        CandleTensor::new(lhs_contiguous.broadcast_matmul(&rhs_contiguous).unwrap())
    }

    fn float_swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        super::base::reshape(tensor, shape)
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.gather(&indices.tensor, dim).unwrap())
    }

    fn float_scatter<const D: usize>(
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

    fn float_select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.index_select(&indices.tensor, dim).unwrap())
    }

    fn float_select_assign<const D: usize>(
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

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        super::base::slice(tensor, ranges)
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        super::base::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where<const D: usize>(
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

    fn float_mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(
            mask.tensor
                .where_cond(
                    &super::candle_utils::fill_like::<F, D>(value, &tensor.tensor),
                    &tensor.tensor,
                )
                .unwrap(),
        )
    }

    fn float_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.eq(&rhs.tensor).unwrap())
    }

    fn float_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .eq(&super::candle_utils::fill_like::<F, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.gt(&rhs.tensor).unwrap())
    }

    fn float_greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .gt(&super::candle_utils::fill_like::<F, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.ge(&rhs.tensor).unwrap())
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .ge(&super::candle_utils::fill_like::<F, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.lt(&rhs.tensor).unwrap())
    }

    fn float_lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .lt(&super::candle_utils::fill_like::<F, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.le(&rhs.tensor).unwrap())
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .le(&super::candle_utils::fill_like::<F, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        let sum = tensor.tensor.sum_all().unwrap().to_scalar::<F>().unwrap();
        CandleTensor::from_data(
            Data::new([sum].into(), [1].into()),
            Self::float_device(&tensor),
        )
    }

    fn float_sum_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.sum_keepdim(dim).unwrap())
    }

    fn float_mean_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.mean_keepdim(dim).unwrap())
    }

    fn float_to_full_precision<const D: usize>(
        tensor: &FloatTensor<Self, D>,
    ) -> FloatTensor<FullPrecisionBackend<Self>, D> {
        CandleTensor::new(tensor.tensor.to_dtype(candle_core::DType::F32).unwrap())
    }

    fn float_from_full_precision<const D: usize>(
        tensor: FloatTensor<FullPrecisionBackend<Self>, D>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.to_dtype(F::DTYPE).unwrap())
    }

    fn float_exp<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.exp().unwrap())
    }

    fn float_log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.log().unwrap())
    }

    fn float_log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new((tensor.tensor + 1.).unwrap().log().unwrap())
    }

    fn float_powf_scalar<const D: usize>(
        tensor: FloatTensor<Self, D>,
        value: f32,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.powf(value.elem::<f64>()).unwrap())
    }

    fn float_sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.sqrt().unwrap())
    }

    fn float_abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.abs().unwrap())
    }

    fn float_cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.cos().unwrap())
    }

    fn float_sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.sin().unwrap())
    }

    fn float_tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.tanh().unwrap())
    }

    fn float_erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.erf().unwrap())
    }

    fn float_cat<const D: usize>(
        tensors: Vec<FloatTensor<Self, D>>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        super::base::cat(tensors, dim)
    }

    fn float_argmax<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .argmax_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn float_argmin<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .argmin_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn float_clamp_max<const D: usize>(
        tensor: FloatTensor<Self, D>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.minimum(max).unwrap())
    }

    fn float_clamp_min<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.maximum(min).unwrap())
    }

    fn float_clamp<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.clamp(min, max).unwrap())
    }

    fn float_recip<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.recip().unwrap())
    }

    fn float_narrow<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> FloatTensor<Self, D> {
        super::base::narrow(tensor, dim, start, length)
    }

    fn float_chunk<const D: usize>(
        tensor: FloatTensor<Self, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<FloatTensor<Self, D>> {
        super::base::chunk(tensor, chunks, dim)
    }

    fn float_powf<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        //broadcast_pow is in main but not yet published
        //note: probably replace once pow once 0.3.3 is out
        //see: https://github.com/huggingface/candle/pull/1583/files#diff-6319fa1e16dadc4c7b4e25698139703d93b70f30a1f8e2ac0999978e39efaa81R2594

        CandleTensor::new(
            rhs.tensor
                .broadcast_mul(&lhs.tensor.log().unwrap())
                .unwrap()
                .exp()
                .unwrap(),
        )
    }

    fn float_permute<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: [usize; D],
    ) -> FloatTensor<Self, D> {
        super::base::permute(tensor, axes)
    }

    fn float_flip<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: &[usize],
    ) -> FloatTensor<Self, D> {
        super::base::flip(tensor, axes)
    }

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        expand(tensor, shape)
    }

    // TODO add sign operator once Candle supports it:
    // https://github.com/huggingface/candle/issues/1827
}
