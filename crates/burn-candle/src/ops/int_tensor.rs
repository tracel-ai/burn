use burn_tensor::{
    ops::{BoolTensor, FloatTensor, IntElem, IntTensor, IntTensorOps},
    Bool, Device, Distribution, ElementConversion, Shape, TensorData,
};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    Candle, CandleTensor,
};

use super::base::{expand, permute, sign};

impl<F: FloatCandleElement, I: IntCandleElement> IntTensorOps<Self> for Candle<F, I> {
    fn int_empty(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        super::base::empty(shape, device, I::DTYPE)
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> TensorData {
        super::base::into_data(tensor)
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        match data.dtype {
            burn_tensor::DType::I64 => super::base::from_data::<i64>(data, device),
            burn_tensor::DType::U32 => super::base::from_data::<u32>(data, device),
            burn_tensor::DType::U8 => super::base::from_data::<u8>(data, device),
            _ => unimplemented!("Unsupported dtype for `int_from_data`"),
        }
    }

    fn int_device(tensor: &IntTensor<Self>) -> Device<Self> {
        super::base::device(tensor)
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &Device<Self>) -> IntTensor<Self> {
        super::base::to_device(tensor, device)
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        super::base::reshape(tensor, shape)
    }

    fn int_slice(tensor: IntTensor<Self>, indices: &[std::ops::Range<usize>]) -> IntTensor<Self> {
        super::base::slice(tensor, indices)
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        indices: &[std::ops::Range<usize>],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        super::base::slice_assign(tensor, indices, value)
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        CandleTensor::new(tensor.tensor.to_dtype(F::DTYPE).unwrap())
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        source: IntTensor<Self>,
    ) -> IntTensor<Self> {
        super::base::mask_where_broadcasted(tensor, mask, source)
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntElem<Self>,
    ) -> IntTensor<Self> {
        CandleTensor::new(
            mask.tensor
                .where_cond(
                    &super::candle_utils::fill_like::<I>(value, &tensor.tensor),
                    &tensor.tensor,
                )
                .unwrap(),
        )
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        CandleTensor::new(tensor.tensor.gather(&indices.tensor, dim).unwrap())
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .scatter_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        CandleTensor::new(tensor.tensor.index_select(&indices.tensor, dim).unwrap())
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .index_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        super::base::cat(tensors, dim)
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.eq(&rhs.tensor).unwrap())
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .eq(&super::candle_utils::fill_like::<I>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.gt(&rhs.tensor).unwrap())
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .gt(&super::candle_utils::fill_like::<I>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.ge(&rhs.tensor).unwrap())
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .ge(&super::candle_utils::fill_like::<I>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.lt(&rhs.tensor).unwrap())
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .lt(&super::candle_utils::fill_like::<I>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.le(&rhs.tensor).unwrap())
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .le(&super::candle_utils::fill_like::<I>(rhs, &lhs.tensor))
                .unwrap(),
        )
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_add(&rhs.tensor).unwrap())
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        CandleTensor::new((lhs.tensor + rhs.elem::<f64>()).unwrap())
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_sub(&rhs.tensor).unwrap())
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        CandleTensor::new((lhs.tensor - rhs.elem::<f64>()).unwrap())
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_mul(&rhs.tensor).unwrap())
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        CandleTensor::new((lhs.tensor * rhs.elem::<f64>()).unwrap())
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        CandleTensor::new(lhs.tensor.broadcast_div(&rhs.tensor).unwrap())
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        // Candle implements scalar a/b as a * (1/b). With ints 1/b is rounded to 0 so we always obtain 0.
        panic!("Not supported by Candle")
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
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

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        // Same problem as int_div_scalar.
        panic!("Not supported by Candle")
    }

    fn int_zeros(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        CandleTensor::new(
            candle_core::Tensor::zeros(shape.dims, I::DTYPE, &(device.clone()).into()).unwrap(),
        )
    }

    fn int_ones(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        CandleTensor::new(
            candle_core::Tensor::ones(shape.dims, I::DTYPE, &(device.clone()).into()).unwrap(),
        )
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let sum = tensor.tensor.sum_all().unwrap().to_scalar::<I>().unwrap();
        CandleTensor::from_data::<I>(
            TensorData::new([sum].into(), [1]),
            Self::int_device(&tensor),
        )
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        CandleTensor::new(tensor.tensor.sum_keepdim(dim).unwrap())
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        todo!("prod is not implemented for Candle IntTensor (see https://github.com/tracel-ai/burn/issues/1454)")
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        todo!("prod_int is not implemented for Candle IntTensor (see https://github.com/tracel-ai/burn/issues/1454)")
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        // Candle implements scalar a/b as a * (1/b). With ints 1/b is rounded to 0 so we always obtain 0.
        panic!("Not supported by Candle")
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .argmax_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .argmin_keepdim(dim)
                .unwrap()
                .to_dtype(I::DTYPE)
                .unwrap(),
        )
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
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

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn int_narrow(
        tensor: IntTensor<Self>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> IntTensor<Self> {
        super::base::narrow(tensor, dim, start, length)
    }

    fn int_chunk(tensor: IntTensor<Self>, chunks: usize, dim: usize) -> Vec<IntTensor<Self>> {
        super::base::chunk(tensor, chunks, dim)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
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

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        super::base::permute(tensor, axes)
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        super::base::flip(tensor, axes)
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        expand(tensor, shape)
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        sign(tensor)
    }
    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_and is not implemented for Candle IntTensor");
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_and_scalar is not implemented for Candle IntTensor");
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_or is not implemented for Candle IntTensor");
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_or_scalar is not implemented for Candle IntTensor");
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_xor is not implemented for Candle IntTensor");
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_xor_scalar is not implemented for Candle IntTensor");
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_not is not implemented for Candle IntTensor");
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_left_shift is not implemented for Candle IntTensor");
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_right_shift is not implemented for Candle IntTensor");
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_left_shift_scalar is not implemented for Candle IntTensor");
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        unimplemented!("bitwise_right_shift_scalar is not implemented for Candle IntTensor");
    }
}
