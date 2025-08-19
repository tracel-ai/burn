use super::TchOps;
use crate::{LibTorch, LibTorchDevice, QuantElement, TchShape, TchTensor, element::TchElement};
use burn_tensor::{Shape, TensorData, TensorMetadata, backend::Backend, ops::BoolTensorOps};
use std::ops::Range;

impl<E: TchElement, Q: QuantElement> BoolTensorOps<Self> for LibTorch<E, Q> {
    fn bool_from_data(data: TensorData, device: &LibTorchDevice) -> TchTensor {
        match data.dtype {
            burn_tensor::DType::Bool => TchTensor::from_data::<bool>(data, (*device).into()),
            _ => unimplemented!("Unsupported dtype for `bool_from_data`"),
        }
    }

    fn bool_repeat_dim(tensor: TchTensor, dim: usize, times: usize) -> TchTensor {
        TchOps::repeat_dim(tensor, dim, times)
    }

    async fn bool_into_data(tensor: TchTensor) -> TensorData {
        let shape = tensor.shape();
        let tensor = Self::bool_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<bool>, tch::TchError> = tensor.tensor.shallow_clone().try_into();
        TensorData::new(values.unwrap(), shape)
    }

    fn bool_to_device(tensor: TchTensor, device: &LibTorchDevice) -> TchTensor {
        TchOps::to_device(tensor, device)
    }

    fn bool_reshape(tensor: TchTensor, shape: Shape) -> TchTensor {
        TchOps::reshape(tensor, shape)
    }

    fn bool_device(tensor: &TchTensor) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn bool_empty(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor {
        let tensor = tch::Tensor::empty(
            TchShape::from(shape).dims,
            (tch::Kind::Bool, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn bool_zeros(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor {
        let tensor = tch::Tensor::zeros(
            TchShape::from(shape).dims,
            (tch::Kind::Bool, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn bool_ones(shape: Shape, device: &<LibTorch<E> as Backend>::Device) -> TchTensor {
        let tensor = tch::Tensor::zeros(
            TchShape::from(shape).dims,
            (tch::Kind::Bool, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn bool_slice(tensor: TchTensor, ranges: &[Range<usize>]) -> TchTensor {
        TchOps::slice(tensor, ranges)
    }

    fn bool_slice_assign(
        tensor: TchTensor,
        ranges: &[Range<usize>],
        value: TchTensor,
    ) -> TchTensor {
        TchOps::slice_assign(tensor, ranges, value)
    }

    fn bool_cat(tensors: Vec<TchTensor>, dim: usize) -> TchTensor {
        TchOps::cat(tensors, dim)
    }

    fn bool_equal(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchOps::equal(lhs, rhs)
    }

    fn bool_not(tensor: TchTensor) -> TchTensor {
        tensor.unary_ops(
            |mut tensor| tensor.eq_(0).to_kind(tch::Kind::Bool),
            |tensor| tensor.eq(0),
        )
    }

    fn bool_and(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.logical_and_(rhs),
            |lhs, rhs| rhs.logical_and_(lhs),
            |lhs, rhs| lhs.logical_and(rhs),
        )
    }

    fn bool_or(lhs: TchTensor, rhs: TchTensor) -> TchTensor {
        TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.logical_or_(rhs),
            |lhs, rhs| rhs.logical_or_(lhs),
            |lhs, rhs| lhs.logical_or(rhs),
        )
    }

    fn bool_into_int(tensor: TchTensor) -> TchTensor {
        let tensor = tensor.tensor.to_kind(tch::Kind::Int64);
        TchTensor::new(tensor)
    }

    fn bool_into_float(tensor: TchTensor) -> TchTensor {
        let tensor = tensor.tensor.to_kind(E::KIND);
        TchTensor::new(tensor)
    }

    fn bool_swap_dims(tensor: TchTensor, dim1: usize, dim2: usize) -> TchTensor {
        TchOps::swap_dims(tensor, dim1, dim2)
    }

    fn bool_permute(tensor: TchTensor, axes: &[usize]) -> TchTensor {
        TchOps::permute(tensor, axes)
    }

    fn bool_flip(tensor: TchTensor, axes: &[usize]) -> TchTensor {
        TchOps::flip(tensor, axes)
    }

    async fn bool_argwhere(tensor: TchTensor) -> TchTensor {
        TchTensor::new(tensor.tensor.argwhere())
    }

    fn bool_expand(tensor: TchTensor, shape: Shape) -> TchTensor {
        TchOps::expand(tensor, shape)
    }
}
