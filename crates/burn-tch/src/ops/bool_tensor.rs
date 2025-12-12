use super::TchOps;
use crate::{LibTorch, LibTorchDevice, TchShape, TchTensor, element::TchElement};
use burn_tensor::ElementConversion;
use burn_tensor::backend::ExecutionError;
use burn_tensor::ops::IntTensor;
use burn_tensor::{Shape, TensorData, TensorMetadata, backend::Backend, ops::BoolTensorOps};

impl<E: TchElement> BoolTensorOps<Self> for LibTorch<E> {
    fn bool_from_data(data: TensorData, device: &LibTorchDevice) -> TchTensor {
        match data.dtype {
            burn_tensor::DType::Bool => TchTensor::from_data::<bool>(data, (*device).into()),
            _ => unimplemented!("Unsupported dtype for `bool_from_data`"),
        }
    }

    fn bool_repeat_dim(tensor: TchTensor, dim: usize, times: usize) -> TchTensor {
        TchOps::repeat_dim(tensor, dim, times)
    }

    async fn bool_into_data(tensor: TchTensor) -> Result<TensorData, ExecutionError> {
        let shape = tensor.shape();
        let tensor = Self::bool_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<bool>, tch::TchError> = tensor.tensor.shallow_clone().try_into();
        Ok(TensorData::new(values.unwrap(), shape))
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
        let tensor = tch::Tensor::ones(
            TchShape::from(shape).dims,
            (tch::Kind::Bool, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn bool_slice(tensor: TchTensor, slices: &[burn_tensor::Slice]) -> TchTensor {
        TchOps::slice_with_steps(tensor, slices)
    }

    fn bool_slice_assign(
        tensor: TchTensor,
        slices: &[burn_tensor::Slice],
        value: TchTensor,
    ) -> TchTensor {
        TchOps::slice_assign(tensor, slices, value)
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
        let tensor = tensor.tensor.to_kind(E::kind());
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

    fn bool_select(tensor: TchTensor, dim: usize, indices: TchTensor) -> TchTensor {
        TchOps::index_select_dim(tensor, dim, indices)
    }

    fn bool_select_or(
        tensor: TchTensor,
        dim: usize,
        indices: TchTensor,
        value: TchTensor,
    ) -> TchTensor {
        TchOps::select_assign(tensor, dim, indices, value)
    }

    fn bool_expand(tensor: TchTensor, shape: Shape) -> TchTensor {
        TchOps::expand(tensor, shape)
    }

    fn bool_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        TchOps::unfold(tensor, dim, size, step)
    }

    fn bool_mask_where(
        tensor: burn_tensor::ops::BoolTensor<Self>,
        mask: burn_tensor::ops::BoolTensor<Self>,
        value: burn_tensor::ops::BoolTensor<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self> {
        TchTensor::binary_ops_tensor(
            tensor,
            value,
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
            |tensor, source| source.f_where_self(&mask.tensor, tensor).unwrap(),
        )
    }

    fn bool_mask_fill(
        tensor: burn_tensor::ops::BoolTensor<Self>,
        mask: burn_tensor::ops::BoolTensor<Self>,
        value: burn_tensor::ops::BoolElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self> {
        tensor.unary_ops(
            |mut tensor| {
                tensor
                    .f_masked_fill_(&mask.tensor, value.elem::<i64>())
                    .unwrap()
            },
            |tensor| {
                tensor
                    .f_masked_fill(&mask.tensor, value.elem::<i64>())
                    .unwrap()
            },
        )
    }

    fn bool_gather(
        dim: usize,
        tensor: burn_tensor::ops::BoolTensor<Self>,
        indices: IntTensor<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self> {
        TchOps::gather(dim, tensor, indices)
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: burn_tensor::ops::BoolTensor<Self>,
        indices: IntTensor<Self>,
        value: burn_tensor::ops::BoolTensor<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self> {
        TchOps::scatter(dim, tensor, indices, value)
    }

    fn bool_equal_elem(
        lhs: burn_tensor::ops::BoolTensor<Self>,
        rhs: burn_tensor::ops::BoolElem<Self>,
    ) -> burn_tensor::ops::BoolTensor<Self> {
        TchOps::equal_elem(lhs, rhs.elem::<i64>())
    }
}
