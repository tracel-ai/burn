use burn_tensor::{
    ops::{BoolTensor, ByteElem, ByteTensor, ByteTensorOps, FloatTensor, IntTensor},
    Bool, Device, Distribution, ElementConversion, Shape, TensorData,
};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleTensor,
};

use super::base::{expand, permute, sign};

impl<F: FloatCandleElement, I: IntCandleElement> ByteTensorOps<Self> for Candle<F, I> {
    fn byte_empty(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        super::base::empty(shape, device, I::DTYPE)
    }

    async fn byte_into_data(tensor: ByteTensor<Self>) -> TensorData {
        super::base::into_data(tensor)
    }

    fn byte_from_data(data: TensorData, device: &Device<Self>) -> ByteTensor<Self> {
        super::base::from_data::<I>(data, device)
    }

    fn byte_device(tensor: &ByteTensor<Self>) -> Device<Self> {
        super::base::device(tensor)
    }

    fn byte_to_device(tensor: ByteTensor<Self>, device: &Device<Self>) -> ByteTensor<Self> {
        super::base::to_device(tensor, device)
    }

    fn byte_reshape(tensor: ByteTensor<Self>, shape: Shape) -> ByteTensor<Self> {
        super::base::reshape(tensor, shape)
    }

    fn byte_slice(
        tensor: ByteTensor<Self>,
        indices: &[std::ops::Range<usize>],
    ) -> ByteTensor<Self> {
        super::base::slice(tensor, indices)
    }

    fn byte_slice_assign(
        tensor: ByteTensor<Self>,
        indices: &[std::ops::Range<usize>],
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        super::base::slice_assign(tensor, indices, value)
    }

    fn byte_into_float(tensor: ByteTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.to_dtype(F::DTYPE).unwrap())
    }

    fn byte_into_int(tensor: ByteTensor<Self>) -> IntTensor<Self> {
        CandleTensor::new(tensor.tensor.to_dtype(I::DTYPE).unwrap())
    }

    fn byte_mask_where(
        tensor: ByteTensor<Self>,
        mask: BoolTensor<Self>,
        source: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        super::base::mask_where_broadcasted(tensor, mask, source)
    }

    fn byte_mask_fill(
        tensor: ByteTensor<Self>,
        mask: BoolTensor<Self>,
        value: ByteElem<Self>,
    ) -> ByteTensor<Self> {
        CandleTensor::new(
            mask.tensor
                .where_cond(
                    &super::candle_utils::fill_like::<u32>(value, &tensor.tensor),
                    &tensor.tensor,
                )
                .unwrap(),
        )
    }

    fn byte_gather(
        dim: usize,
        tensor: ByteTensor<Self>,
        indices: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        CandleTensor::new(tensor.tensor.gather(&indices.tensor, dim).unwrap())
    }

    fn byte_scatter(
        dim: usize,
        tensor: ByteTensor<Self>,
        indices: ByteTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .scatter_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn byte_select(
        tensor: ByteTensor<Self>,
        dim: usize,
        indices: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        CandleTensor::new(tensor.tensor.index_select(&indices.tensor, dim).unwrap())
    }

    fn byte_select_assign(
        tensor: ByteTensor<Self>,
        dim: usize,
        indices: ByteTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .index_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn byte_cat(tensors: Vec<ByteTensor<Self>>, dim: usize) -> ByteTensor<Self> {
        super::base::cat(tensors, dim)
    }

    fn byte_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.eq(&rhs.tensor).unwrap())
    }

    fn byte_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .eq(&super::candle_utils::fill_like::<u32>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn byte_greater(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.gt(&rhs.tensor).unwrap())
    }

    fn byte_greater_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .gt(&super::candle_utils::fill_like::<u32>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn byte_greater_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.ge(&rhs.tensor).unwrap())
    }

    fn byte_greater_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .ge(&super::candle_utils::fill_like::<u32>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn byte_lower(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.lt(&rhs.tensor).unwrap())
    }

    fn byte_lower_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .lt(&super::candle_utils::fill_like::<u32>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn byte_lower_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.le(&rhs.tensor).unwrap())
    }

    fn byte_lower_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .le(&super::candle_utils::fill_like::<u32>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn byte_add(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_add(&rhs.tensor).unwrap())
    }

    fn byte_add_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        CandleTensor::new((lhs.tensor + rhs.elem::<f64>()).unwrap())
    }

    fn byte_sub(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_sub(&rhs.tensor).unwrap())
    }

    fn byte_sub_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        CandleTensor::new((lhs.tensor - rhs.elem::<f64>()).unwrap())
    }

    fn byte_mul(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_mul(&rhs.tensor).unwrap())
    }

    fn byte_mul_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        CandleTensor::new((lhs.tensor * rhs.elem::<f64>()).unwrap())
    }

    fn byte_div(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_div(&rhs.tensor).unwrap())
    }

    fn byte_div_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        // Candle implements scalar a/b as a * (1/b). With ints 1/b is rounded to 0 so we always obtain 0.
        panic!("Not supported by Candle")
    }

    fn byte_remainder(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        CandleTensor::new(
            (lhs.tensor.clone()
                - lhs
                    .tensor
                    .broadcast_div(&rhs.tensor)
                    .unwrap()
                    .broadcast_mul(&rhs.tensor)
                    .unwrap())
            .unwrap(),
        )
    }

    fn byte_remainder_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        // Same problem as int_div_scalar.
        panic!("Not supported by Candle")
    }

    fn byte_zeros(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        CandleTensor::new(
            candle_core::Tensor::zeros(shape.dims, I::DTYPE, &(device.clone()).into()).unwrap(),
        )
    }

    fn byte_ones(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        CandleTensor::new(
            candle_core::Tensor::ones(shape.dims, I::DTYPE, &(device.clone()).into()).unwrap(),
        )
    }

    fn byte_sum(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        let sum = tensor.tensor.sum_all().unwrap().to_scalar::<I>().unwrap();
        CandleTensor::from_data::<I>(
            TensorData::new([sum].into(), [1]),
            Self::byte_device(&tensor),
        )
    }

    fn byte_sum_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        CandleTensor::new(tensor.tensor.sum_keepdim(dim).unwrap())
    }

    fn byte_prod(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        todo!("prod is not implemented for Candle ByteTensor (see https://github.com/tracel-ai/burn/issues/1454)")
    }

    fn byte_prod_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        todo!("prod_int is not implemented for Candle ByteTensor (see https://github.com/tracel-ai/burn/issues/1454)")
    }

    fn byte_mean_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        // Candle implements scalar a/b as a * (1/b). With ints 1/b is rounded to 0 so we always obtain 0.
        panic!("Not supported by Candle")
    }

    fn byte_argmax(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .argmax_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn byte_argmin(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .argmin_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn byte_abs(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
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

    fn byte_swap_dims(tensor: ByteTensor<Self>, dim1: usize, dim2: usize) -> ByteTensor<Self> {
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn byte_narrow(
        tensor: ByteTensor<Self>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> ByteTensor<Self> {
        super::base::narrow(tensor, dim, start, length)
    }

    fn byte_chunk(tensor: ByteTensor<Self>, chunks: usize, dim: usize) -> Vec<ByteTensor<Self>> {
        super::base::chunk(tensor, chunks, dim)
    }

    fn byte_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> ByteTensor<Self> {
        let shape = shape.dims;
        let device = &(device.clone()).into();
        match distribution {
            Distribution::Default => CandleTensor::new(
                candle_core::Tensor::rand(0.elem::<F>(), 255.elem::<F>(), shape, device)
                    .unwrap()
                    .to_dtype(I::DTYPE)
                    .unwrap(),
            ),
            Distribution::Bernoulli(prob) => CandleTensor::new(
                candle_core::Tensor::rand(0.elem::<F>(), 1.elem::<F>(), shape.clone(), device)
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

    fn byte_permute(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        super::base::permute(tensor, axes)
    }

    fn byte_flip(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        super::base::flip(tensor, axes)
    }

    fn byte_expand(tensor: ByteTensor<Self>, shape: Shape) -> ByteTensor<Self> {
        expand(tensor, shape)
    }

    fn byte_sign(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        sign(tensor)
    }
}
