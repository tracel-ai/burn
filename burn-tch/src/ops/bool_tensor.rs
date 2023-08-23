use std::ops::Range;

use burn_tensor::{backend::Backend, ops::BoolTensorOps, Data, Shape};

use crate::{element::TchElement, TchBackend, TchDevice, TchTensor};

use super::TchOps;

impl<E: TchElement> BoolTensorOps<TchBackend<E>> for TchBackend<E> {
    fn bool_from_data<const D: usize>(
        data: Data<bool, D>,
        device: &TchDevice,
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

    fn bool_to_data<const D: usize>(tensor: &TchTensor<bool, D>) -> Data<bool, D> {
        let shape = Self::bool_shape(tensor);
        let tensor = Self::bool_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let values: Result<Vec<bool>, tch::TchError> = tensor.tensor.shallow_clone().try_into();

        Data::new(values.unwrap(), shape)
    }

    fn bool_into_data<const D: usize>(tensor: TchTensor<bool, D>) -> Data<bool, D> {
        Self::bool_to_data(&tensor)
    }

    fn bool_to_device<const D: usize>(
        tensor: TchTensor<bool, D>,
        device: &TchDevice,
    ) -> TchTensor<bool, D> {
        TchTensor::new(tensor.tensor.to((*device).into()))
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: TchTensor<bool, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<bool, D2> {
        TchOps::reshape(tensor, shape)
    }

    fn bool_device<const D: usize>(tensor: &TchTensor<bool, D>) -> TchDevice {
        tensor.tensor.device().into()
    }

    fn bool_empty<const D: usize>(
        shape: Shape<D>,
        device: &<TchBackend<E> as Backend>::Device,
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

    fn bool_equal_elem<const D: usize>(lhs: TchTensor<bool, D>, rhs: bool) -> TchTensor<bool, D> {
        let rhs = match rhs {
            true => 1,
            false => 0,
        };

        lhs.unary_ops(
            |mut tensor| tensor.eq_(rhs).to_kind(tch::Kind::Bool),
            |tensor| tensor.eq(rhs),
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
}
