use super::TchOps;
use crate::{element::TchElement, LibTorch, LibTorchDevice, TchTensor};
use burn_tensor::{backend::Backend, ops::BoolTensorOps, Data, Reader, Shape};
use std::ops::Range;

impl<E: TchElement> BoolTensorOps<Self> for LibTorch<E> {
    fn bool_from_data<const D: usize>(
        data: Data<bool, D>,
        device: &LibTorchDevice,
    ) -> TchTensor<bool, D> {
        TchTensor::from_data(data, (*device).into())
    }

    fn bool_shape<const D: usize>(tensor: &TchTensor<bool, D>) -> Shape<D> {
        tensor.shape()
    }

    fn bool_repeat<const D: usize>(
        tensor: TchTensor<bool, D>,
        dim: usize,
        times: usize,
    ) -> TchTensor<bool, D> {
        TchOps::repeat(tensor, dim, times)
    }

    fn bool_into_data<const D: usize>(tensor: TchTensor<bool, D>) -> Reader<Data<bool, D>> {
        let shape = Self::bool_shape(&tensor);
        let tensor = Self::bool_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<bool>, tch::TchError> = tensor.tensor.shallow_clone().try_into();

        Reader::Concrete(Data::new(values.unwrap(), shape))
    }

    fn bool_to_device<const D: usize>(
        tensor: TchTensor<bool, D>,
        device: &LibTorchDevice,
    ) -> TchTensor<bool, D> {
        TchOps::to_device(tensor, device)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: TchTensor<bool, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<bool, D2> {
        TchOps::reshape(tensor, shape)
    }

    fn bool_device<const D: usize>(tensor: &TchTensor<bool, D>) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn bool_empty<const D: usize>(
        shape: Shape<D>,
        device: &<LibTorch<E> as Backend>::Device,
    ) -> TchTensor<bool, D> {
        let tensor = tch::Tensor::empty(
            shape.dims.map(|a| a as i64),
            (tch::Kind::Bool, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: TchTensor<bool, D1>,
        ranges: [Range<usize>; D2],
    ) -> TchTensor<bool, D1> {
        TchOps::slice(tensor, ranges)
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: TchTensor<bool, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: TchTensor<bool, D1>,
    ) -> TchTensor<bool, D1> {
        TchOps::slice_assign(tensor, ranges, value)
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<TchTensor<bool, D>>,
        dim: usize,
    ) -> TchTensor<bool, D> {
        TchOps::cat(tensors, dim)
    }

    fn bool_equal<const D: usize>(
        lhs: TchTensor<bool, D>,
        rhs: TchTensor<bool, D>,
    ) -> TchTensor<bool, D> {
        TchOps::equal(lhs, rhs)
    }

    fn bool_not<const D: usize>(tensor: TchTensor<bool, D>) -> TchTensor<bool, D> {
        tensor.unary_ops(
            |mut tensor| tensor.eq_(0).to_kind(tch::Kind::Bool),
            |tensor| tensor.eq(0),
        )
    }

    fn bool_into_int<const D: usize>(tensor: TchTensor<bool, D>) -> TchTensor<i64, D> {
        let tensor = tensor.tensor.to_kind(tch::Kind::Int64);
        TchTensor::new(tensor)
    }

    fn bool_into_float<const D: usize>(tensor: TchTensor<bool, D>) -> TchTensor<E, D> {
        let tensor = tensor.tensor.to_kind(E::KIND);
        TchTensor::new(tensor)
    }

    fn bool_swap_dims<const D: usize>(
        tensor: <LibTorch<E> as Backend>::BoolTensorPrimitive<D>,
        dim1: usize,
        dim2: usize,
    ) -> <LibTorch<E> as Backend>::BoolTensorPrimitive<D> {
        TchOps::swap_dims(tensor, dim1, dim2)
    }

    fn bool_narrow<const D: usize>(
        tensor: TchTensor<bool, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> TchTensor<bool, D> {
        TchOps::narrow(tensor, dim, start, length)
    }

    fn bool_chunk<const D: usize>(
        tensor: TchTensor<bool, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<TchTensor<bool, D>> {
        TchOps::chunk(tensor, chunks, dim)
    }

    fn bool_permute<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<Self, D>,
        axes: [usize; D],
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        TchOps::permute(tensor, axes)
    }

    fn bool_flip<const D: usize>(tensor: TchTensor<bool, D>, axes: &[usize]) -> TchTensor<bool, D> {
        TchOps::flip(tensor, axes)
    }

    fn bool_argwhere<const D: usize>(
        tensor: <LibTorch<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> TchTensor<i64, 2> {
        TchTensor::new(tensor.tensor.argwhere())
    }

    fn bool_nonzero<const D: usize>(
        tensor: <LibTorch<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Vec<TchTensor<i64, 1>> {
        tensor
            .tensor
            .nonzero_numpy()
            .into_iter()
            .map(TchTensor::new)
            .collect()
    }

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> burn_tensor::ops::BoolTensor<Self, D2> {
        TchOps::expand(tensor, shape)
    }
}
