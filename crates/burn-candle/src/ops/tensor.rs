use std::borrow::Borrow;

use burn_tensor::{
    ops::{BoolTensor, FloatElem, FloatTensor, FloatTensorOps, FullPrecisionBackend, IntTensor},
    Device, Distribution, ElementConversion, Shape, TensorData,
};
use candle_core::{backend::BackendStorage, shape, Tensor};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleTensor,
};

use super::base::{expand, permute, sign};

impl<F: FloatCandleElement, I: IntCandleElement> FloatTensorOps<Self> for Candle<F, I> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> CandleTensor<F> {
        CandleTensor::from_data(data, device.clone())
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        let shape = shape.dims;
        let device = &(device.clone()).into();
        match distribution {
            Distribution::Default => CandleTensor::new(
                candle_core::Tensor::rand(0.elem::<F>(), 1.elem::<F>(), shape, device)
                    .unwrap()
                    .to_dtype(F::DTYPE)
                    .unwrap(),
            ),
            Distribution::Bernoulli(prob) => CandleTensor::new(
                candle_core::Tensor::rand(0.elem::<F>(), 1.elem::<F>(), shape.clone(), device)
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

    fn float_shape(tensor: &CandleTensor<F>) -> Shape {
        super::base::shape(tensor)
    }

    async fn float_into_data(tensor: CandleTensor<F>) -> TensorData {
        super::base::into_data(tensor)
    }

    fn float_device(tensor: &CandleTensor<F>) -> Device<Self> {
        super::base::device(tensor)
    }

    fn float_to_device(tensor: CandleTensor<F>, device: &Device<Self>) -> CandleTensor<F> {
        super::base::to_device(tensor, device)
    }

    fn float_into_int(tensor: CandleTensor<F>) -> IntTensor<Self> {
        CandleTensor::new(tensor.tensor.to_dtype(I::DTYPE).unwrap())
    }

    fn float_empty(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        super::base::empty(shape, device)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_add(&rhs.tensor).unwrap())
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        CandleTensor::new((lhs.tensor + rhs.elem::<f64>()).unwrap())
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_sub(&rhs.tensor).unwrap())
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        CandleTensor::new((lhs.tensor - rhs.elem::<f64>()).unwrap())
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_mul(&rhs.tensor).unwrap())
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        CandleTensor::new((lhs.tensor * rhs.elem::<f64>()).unwrap())
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_div(&rhs.tensor).unwrap())
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        CandleTensor::new((lhs.tensor / rhs.elem::<f64>()).unwrap())
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        // In PyTorch, remainder can also be defined as torch.remainder(a, b) == a - a.div(b, rounding_mode="floor") * b
        let rhs_val = rhs.elem::<f64>();
        let division_result = (lhs.tensor.clone() / rhs_val).unwrap().floor().unwrap();
        let product = division_result * rhs_val;

        CandleTensor::new((lhs.tensor - product).unwrap())
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
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

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        super::base::reshape(tensor, shape)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.gather(&indices.tensor, dim).unwrap())
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .scatter_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.index_select(&indices.tensor, dim).unwrap())
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .index_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn float_slice(
        tensor: FloatTensor<Self>,
        ranges: &[std::ops::Range<usize>],
    ) -> FloatTensor<Self> {
        super::base::slice(tensor, ranges)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[std::ops::Range<usize>],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        super::base::slice_assign(tensor, ranges, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        CandleTensor::new(
            mask.tensor
                .where_cond(&value.tensor, &tensor.tensor)
                .unwrap(),
        )
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        CandleTensor::new(
            mask.tensor
                .where_cond(
                    &super::candle_utils::fill_like::<F>(value, &tensor.tensor),
                    &tensor.tensor,
                )
                .unwrap(),
        )
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.eq(&rhs.tensor).unwrap())
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .eq(&super::candle_utils::fill_like::<F>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.gt(&rhs.tensor).unwrap())
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .gt(&super::candle_utils::fill_like::<F>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.ge(&rhs.tensor).unwrap())
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .ge(&super::candle_utils::fill_like::<F>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.lt(&rhs.tensor).unwrap())
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .lt(&super::candle_utils::fill_like::<F>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.le(&rhs.tensor).unwrap())
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .le(&super::candle_utils::fill_like::<F>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let sum = tensor.tensor.sum_all().unwrap().to_scalar::<F>().unwrap();
        CandleTensor::from_data(
            TensorData::new([sum].into(), [1]),
            Self::float_device(&tensor),
        )
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.sum_keepdim(dim).unwrap())
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.mean_keepdim(dim).unwrap())
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.exp().unwrap())
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.log().unwrap())
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new((tensor.tensor + 1.).unwrap().log().unwrap())
    }

    fn float_powf_scalar(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.powf(value.elem::<f64>()).unwrap())
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.sqrt().unwrap())
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.abs().unwrap())
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.cos().unwrap())
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.sin().unwrap())
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.tanh().unwrap())
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.erf().unwrap())
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        super::base::cat(tensors, dim)
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .argmax_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .argmin_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: FloatElem<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.minimum(max).unwrap())
    }

    fn float_clamp_min(tensor: FloatTensor<Self>, min: FloatElem<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.maximum(min).unwrap())
    }

    fn float_clamp(
        tensor: FloatTensor<Self>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.clamp(min, max).unwrap())
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.recip().unwrap())
    }

    fn float_narrow(
        tensor: FloatTensor<Self>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> FloatTensor<Self> {
        super::base::narrow(tensor, dim, start, length)
    }

    fn float_chunk(tensor: FloatTensor<Self>, chunks: usize, dim: usize) -> Vec<FloatTensor<Self>> {
        super::base::chunk(tensor, chunks, dim)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
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

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        super::base::permute(tensor, axes)
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        super::base::flip(tensor, axes)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        expand(tensor, shape)
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        sign(tensor)
    }
}
