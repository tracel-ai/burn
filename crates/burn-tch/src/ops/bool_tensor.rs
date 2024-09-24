use super::TchOps;
use crate::{element::TchElement, LibTorch, LibTorchDevice, QuantElement, TchShape, TchTensor};
use burn_tensor::{backend::Backend, ops::BoolTensorOps, Shape, TensorData};
use std::ops::Range;

impl<E: TchElement, Q: QuantElement> BoolTensorOps<Self> for LibTorch<E, Q> {
    fn bool_from_data(data: TensorData, device: &LibTorchDevice) -> TchTensor<bool> {
        TchTensor::from_data(data, (*device).into())
    }

    fn bool_shape(tensor: &TchTensor<bool>) -> Shape {
        tensor.shape()
    }

    fn bool_repeat_dim(tensor: TchTensor<bool>, dim: usize, times: usize) -> TchTensor<bool> {
        TchOps::repeat_dim(tensor, dim, times)
    }

    async fn bool_into_data(tensor: TchTensor<bool>) -> TensorData {
        let shape = Self::bool_shape(&tensor);
        let tensor = Self::bool_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<bool>, tch::TchError> = tensor.tensor.shallow_clone().try_into();
        TensorData::new(values.unwrap(), shape)
    }

    fn bool_to_device(tensor: TchTensor<bool>, device: &LibTorchDevice) -> TchTensor<bool> {
        TchOps::to_device(tensor, device)
    }

    fn bool_reshape(tensor: TchTensor<bool>, shape: Shape) -> TchTensor<bool> {
        TchOps::reshape(tensor, shape)
    }

    fn bool_device(tensor: &TchTensor<bool>) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn bool_empty(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor<bool> {
        let tensor = tch::Tensor::empty(
            TchShape::from(shape).dims,
            (tch::Kind::Bool, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn bool_slice(tensor: TchTensor<bool>, ranges: &[Range<usize>]) -> TchTensor<bool> {
        TchOps::slice(tensor, ranges)
    }

    fn bool_slice_assign(
        tensor: TchTensor<bool>,
        ranges: &[Range<usize>],
        value: TchTensor<bool>,
    ) -> TchTensor<bool> {
        TchOps::slice_assign(tensor, ranges, value)
    }

    fn bool_cat(tensors: Vec<TchTensor<bool>>, dim: usize) -> TchTensor<bool> {
        TchOps::cat(tensors, dim)
    }

    fn bool_equal(lhs: TchTensor<bool>, rhs: TchTensor<bool>) -> TchTensor<bool> {
        TchOps::equal(lhs, rhs)
    }

    fn bool_not(tensor: TchTensor<bool>) -> TchTensor<bool> {
        tensor.unary_ops(
            |mut tensor| tensor.eq_(0).to_kind(tch::Kind::Bool),
            |tensor| tensor.eq(0),
        )
    }

    fn bool_into_int(tensor: TchTensor<bool>) -> TchTensor<i64> {
        let tensor = tensor.tensor.to_kind(tch::Kind::Int64);
        TchTensor::new(tensor)
    }

    fn bool_into_float(tensor: TchTensor<bool>) -> TchTensor<E> {
        let tensor = tensor.tensor.to_kind(E::KIND);
        TchTensor::new(tensor)
    }

    fn bool_swap_dims(tensor: TchTensor<bool>, dim1: usize, dim2: usize) -> TchTensor<bool> {
        TchOps::swap_dims(tensor, dim1, dim2)
    }

    fn bool_narrow(
        tensor: TchTensor<bool>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> TchTensor<bool> {
        TchOps::narrow(tensor, dim, start, length)
    }

    fn bool_chunk(tensor: TchTensor<bool>, chunks: usize, dim: usize) -> Vec<TchTensor<bool>> {
        TchOps::chunk(tensor, chunks, dim)
    }

    fn bool_permute(tensor: TchTensor<bool>, axes: &[usize]) -> TchTensor<bool> {
        TchOps::permute(tensor, axes)
    }

    fn bool_flip(tensor: TchTensor<bool>, axes: &[usize]) -> TchTensor<bool> {
        TchOps::flip(tensor, axes)
    }

    async fn bool_argwhere(tensor: TchTensor<bool>) -> TchTensor<i64> {
        TchTensor::new(tensor.tensor.argwhere())
    }

    async fn bool_nonzero(tensor: TchTensor<bool>) -> Vec<TchTensor<i64>> {
        tensor
            .tensor
            .nonzero_numpy()
            .into_iter()
            // As opposed to tch, the resulting vector should be empty for zero tensors
            .filter_map(|t| if t.numel() > 0 { Some(t) } else { None })
            .map(TchTensor::new)
            .collect()
    }

    fn bool_expand(tensor: TchTensor<bool>, shape: Shape) -> TchTensor<bool> {
        TchOps::expand(tensor, shape)
    }
}
