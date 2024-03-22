use burn_tensor::{
    ops::{BoolTensor, FloatTensor, IntElem, IntTensor, IntTensorOps},
    Bool, Data, Device, Distribution, ElementConversion, Reader, Shape,
};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleTensor,
};

use super::base::{expand, permute};

impl<F: FloatCandleElement, I: IntCandleElement> IntTensorOps<Self> for Candle<F, I> {
    fn int_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        super::base::empty(shape, device)
    }

    fn int_shape<const D: usize>(tensor: &IntTensor<Self, D>) -> Shape<D> {
        super::base::shape(tensor)
    }

    fn int_into_data<const D: usize>(tensor: IntTensor<Self, D>) -> Reader<Data<IntElem<Self>, D>> {
        Reader::Concrete(super::base::into_data(tensor))
    }

    fn int_from_data<const D: usize>(
        data: Data<IntElem<Self>, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        super::base::from_data(data, device)
    }

    fn int_device<const D: usize>(tensor: &IntTensor<Self, D>) -> Device<Self> {
        super::base::device(tensor)
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<Self, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        super::base::to_device(tensor, device)
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        super::base::reshape(tensor, shape)
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> IntTensor<Self, D1> {
        super::base::slice(tensor, indices)
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        indices: [std::ops::Range<usize>; D2],
        value: IntTensor<Self, D1>,
    ) -> IntTensor<Self, D1> {
        super::base::slice_assign(tensor, indices, value)
    }

    fn int_into_float<const D: usize>(tensor: IntTensor<Self, D>) -> FloatTensor<Self, D> {
        CandleTensor::new(tensor.tensor.to_dtype(F::DTYPE).unwrap())
    }

    fn int_mask_where<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        source: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(
            mask.tensor
                .where_cond(&source.tensor, &tensor.tensor)
                .unwrap(),
        )
    }

    fn int_mask_fill<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(
            mask.tensor
                .where_cond(
                    &super::candle_utils::fill_like::<I, D>(value, &tensor.tensor),
                    &tensor.tensor,
                )
                .unwrap(),
        )
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(tensor.tensor.gather(&indices.tensor, dim).unwrap())
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .scatter_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn int_select<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(tensor.tensor.index_select(&indices.tensor, dim).unwrap())
    }

    fn int_select_assign<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .index_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn int_cat<const D: usize>(tensors: Vec<IntTensor<Self, D>>, dim: usize) -> IntTensor<Self, D> {
        super::base::cat(tensors, dim)
    }

    fn int_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.eq(&rhs.tensor).unwrap())
    }

    fn int_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .eq(&super::candle_utils::fill_like::<I, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_greater<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.gt(&rhs.tensor).unwrap())
    }

    fn int_greater_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .gt(&super::candle_utils::fill_like::<I, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_greater_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.ge(&rhs.tensor).unwrap())
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .ge(&super::candle_utils::fill_like::<I, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_lower<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.lt(&rhs.tensor).unwrap())
    }

    fn int_lower_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .lt(&super::candle_utils::fill_like::<I, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_lower_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(lhs.tensor.le(&rhs.tensor).unwrap())
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        CandleTensor::new(
            lhs.tensor
                .le(&super::candle_utils::fill_like::<I, D>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_add<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_add(&rhs.tensor).unwrap())
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new((lhs.tensor + rhs.elem::<f64>()).unwrap())
    }

    fn int_sub<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_sub(&rhs.tensor).unwrap())
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new((lhs.tensor - rhs.elem::<f64>()).unwrap())
    }

    fn int_mul<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_mul(&rhs.tensor).unwrap())
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new((lhs.tensor * rhs.elem::<f64>()).unwrap())
    }

    fn int_div<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        CandleTensor::new(lhs.tensor.broadcast_div(&rhs.tensor).unwrap())
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        // Candle implements scalar a/b as a * (1/b). With ints 1/b is rounded to 0 so we always obtain 0.
        panic!("Not supported by Candle")
    }

    fn int_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        CandleTensor::new(
            candle_core::Tensor::zeros(&shape.dims, I::DTYPE, &(*device).into()).unwrap(),
        )
    }

    fn int_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        CandleTensor::new(
            candle_core::Tensor::ones(&shape.dims, I::DTYPE, &(*device).into()).unwrap(),
        )
    }

    fn int_sum<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        let sum = tensor.tensor.sum_all().unwrap().to_scalar::<I>().unwrap();
        CandleTensor::from_data(
            Data::new([sum].into(), [1].into()),
            Self::int_device(&tensor),
        )
    }

    fn int_sum_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        CandleTensor::new(tensor.tensor.sum_keepdim(dim).unwrap())
    }

    fn int_prod<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        todo!("prod is not implemented for Candle IntTensor (see https://github.com/tracel-ai/burn/issues/1454)")
    }

    fn int_prod_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        todo!("prod_int is not implemented for Candle IntTensor (see https://github.com/tracel-ai/burn/issues/1454)")
    }

    fn int_mean_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        // Candle implements scalar a/b as a * (1/b). With ints 1/b is rounded to 0 so we always obtain 0.
        panic!("Not supported by Candle")
    }

    fn int_argmax<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .argmax_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn int_argmin<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        CandleTensor::new(
            tensor
                .tensor
                .argmin_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn int_abs<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, D> {
        // Ugly type conversion here as Candle does not support unary ops on ints
        CandleTensor::new(
            tensor
                .tensor
                .to_dtype(F::DTYPE)
                .unwrap()
                .abs()
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn int_swap_dims<const D: usize>(
        tensor: <Candle<F, I> as burn_tensor::backend::Backend>::IntTensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <Candle<F, I> as burn_tensor::backend::Backend>::IntTensorPrimitive<D> {
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn int_narrow<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> IntTensor<Self, D> {
        super::base::narrow(tensor, dim, start, length)
    }

    fn int_chunk<const D: usize>(
        tensor: IntTensor<Self, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<IntTensor<Self, D>> {
        super::base::chunk(tensor, chunks, dim)
    }

    fn int_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        let shape = &shape.dims;
        let device = &(*device).into();
        match distribution {
            Distribution::Default => CandleTensor::new(
                candle_core::Tensor::rand(0.elem::<F>(), 255.elem::<F>(), shape, device)
                    .unwrap()
                    .to_dtype(I::DTYPE)
                    .unwrap(),
            ),
            Distribution::Bernoulli(prob) => CandleTensor::new(
                candle_core::Tensor::rand(0.elem::<F>(), 1.elem::<F>(), shape, device)
                    .unwrap()
                    .to_dtype(I::DTYPE)
                    .unwrap()
                    .lt(&super::candle_utils::fill(prob, shape, I::DTYPE, device))
                    .unwrap()
                    .to_dtype(I::DTYPE)
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

    fn int_permute<const D: usize>(
        tensor: IntTensor<Self, D>,
        axes: [usize; D],
    ) -> IntTensor<Self, D> {
        super::base::permute(tensor, axes)
    }

    fn int_flip<const D: usize>(tensor: IntTensor<Self, D>, axes: &[usize]) -> IntTensor<Self, D> {
        super::base::flip(tensor, axes)
    }

    fn int_expand<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        expand(tensor, shape)
    }

    // TODO add sign operator once Candle supports it:
    // https://github.com/huggingface/candle/issues/1827
}
