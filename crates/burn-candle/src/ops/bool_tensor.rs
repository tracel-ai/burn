use burn_backend::{
    BackTrace, DType, ExecutionError, Scalar, Shape, Slice, TensorData, TensorMetadata,
    ops::BoolTensorOps,
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};

use crate::{
    Candle, CandleTensor,
    element::{CandleElement, FloatCandleElement, IntCandleElement},
};

use super::base::{expand, permute, unfold};

impl<F: FloatCandleElement, I: IntCandleElement> BoolTensorOps<Self> for Candle<F, I> {
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::base::empty(shape, device, candle_core::DType::U8)
    }

    fn bool_zeros(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::base::zeros(shape, device, candle_core::DType::U8)
    }

    fn bool_ones(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        super::base::ones(shape, device, candle_core::DType::U8)
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> Result<TensorData, ExecutionError> {
        let x: Vec<u8> = tensor
            .tensor
            .flatten_all()
            .map_err(|err| ExecutionError::Generic {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?
            .to_vec1()
            .map_err(|err| ExecutionError::Generic {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?;

        let y = x.iter().map(|b| !matches!(b, 0)).collect();

        Ok(TensorData::new(y, tensor.shape()))
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        match data.dtype {
            DType::U8 => super::base::from_data::<u8>(data, device),
            _ => unimplemented!("Unsupported dtype for `bool_from_data`"),
        }
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

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        super::base::slice_with_steps(tensor, slices)
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        super::base::slice_assign(tensor, slices, value)
    }

    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        super::base::cat(tensors, dim)
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let (lhs_broadcast, rhs_broadcast) =
            super::candle_utils::broadcast_for_comparison(&lhs.tensor, &rhs.tensor).unwrap();
        CandleTensor::new(lhs_broadcast.eq(&rhs_broadcast).unwrap())
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let x = (candle_core::Tensor::zeros_like(&tensor.tensor).unwrap());
        CandleTensor::new(tensor.tensor.eq(&x).unwrap())
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let x = candle_core::Tensor::ones_like(&lhs.tensor).unwrap();
        CandleTensor::new(lhs.tensor.add(&rhs.tensor).unwrap().gt(&x).unwrap())
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        CandleTensor::new(
            lhs.tensor
                .add(&rhs.tensor)
                .unwrap()
                .clamp(0u32, 1u32)
                .unwrap(),
        )
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        super::base::swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        super::base::permute(tensor, axes)
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        super::base::flip(tensor, axes)
    }

    fn bool_select(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        CandleTensor::new(tensor.tensor.index_select(&indices.tensor, dim).unwrap())
    }

    fn bool_select_or(
        tensor: BoolTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .index_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        expand(tensor, shape)
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        unfold(tensor, dim, size, step)
    }

    fn bool_mask_where(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        super::base::mask_where_broadcasted(tensor, mask, value)
    }

    fn bool_mask_fill(
        tensor: BoolTensor<Self>,
        mask: BoolTensor<Self>,
        value: Scalar,
    ) -> BoolTensor<Self> {
        CandleTensor::new(
            mask.tensor
                .where_cond(
                    &super::candle_utils::fill_like::<u8>(value.elem(), &tensor.tensor),
                    &tensor.tensor,
                )
                .unwrap(),
        )
    }

    fn bool_gather(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> BoolTensor<Self> {
        let tensor = tensor.tensor.contiguous().unwrap();
        let indices = indices.tensor.contiguous().unwrap();
        CandleTensor::new(tensor.gather(&indices, dim).unwrap())
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        CandleTensor::new(
            tensor
                .tensor
                .scatter_add(&indices.tensor, &value.tensor, dim)
                .unwrap(),
        )
    }

    fn bool_equal_elem(lhs: BoolTensor<Self>, rhs: Scalar) -> BoolTensor<Self> {
        CandleTensor::new(lhs.tensor.eq(rhs.elem::<u8>()).unwrap())
    }
}
