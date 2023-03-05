use std::{ops::Range, sync::Arc};

use burn_tensor::{backend::Backend, ops::BoolTensorOps, Data, Shape};

use crate::{element::TchElement, to_tensor, TchBackend, TchDevice, TchKind, TchShape, TchTensor};

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

    fn bool_to_data<const D: usize>(tensor: &TchTensor<bool, D>) -> Data<bool, D> {
        let values: Vec<bool> = tensor.tensor.shallow_clone().into();
        Data::new(values, tensor.shape())
    }

    fn bool_into_data<const D: usize>(tensor: TchTensor<bool, D>) -> Data<bool, D> {
        let shape = tensor.shape();
        let values: Vec<bool> = tensor.unary_ops(
            |tensor| tensor.into(),
            |tensor| tensor.shallow_clone().into(),
        );
        Data::new(values, shape)
    }

    fn bool_to_device<const D: usize>(
        tensor: TchTensor<bool, D>,
        device: &TchDevice,
    ) -> TchTensor<bool, D> {
        TchTensor {
            kind: tensor.kind,
            tensor: Arc::new(tensor.tensor.to((*device).into())),
        }
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: TchTensor<bool, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<bool, D2> {
        let shape_tch: TchShape<D2> = shape.into();
        let tensor = tensor.unary_ops(
            |mut tensor| tensor.resize_(&shape_tch.dims),
            |tensor| tensor.reshape(&shape_tch.dims),
        );

        TchTensor {
            tensor: Arc::new(tensor),
            kind: TchKind::new(),
        }
    }

    fn bool_device<const D: usize>(tensor: &TchTensor<bool, D>) -> TchDevice {
        tensor.tensor.device().into()
    }

    fn bool_empty<const D: usize>(
        shape: Shape<D>,
        device: &<TchBackend<E> as Backend>::Device,
    ) -> TchTensor<bool, D> {
        let tensor = tch::Tensor::empty(
            &shape.dims.map(|a| a as i64),
            (tch::Kind::Bool, (*device).into()),
        );

        to_tensor(tensor)
    }

    fn bool_index<const D1: usize, const D2: usize>(
        tensor: TchTensor<bool, D1>,
        indexes: [Range<usize>; D2],
    ) -> TchTensor<bool, D1> {
        TchOps::index(tensor, indexes)
    }
    fn bool_index_assign<const D1: usize, const D2: usize>(
        tensor: TchTensor<bool, D1>,
        indexes: [std::ops::Range<usize>; D2],
        value: TchTensor<bool, D1>,
    ) -> TchTensor<bool, D1> {
        TchOps::index_assign(tensor, indexes, value)
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
        let tensor = lhs.unary_ops(
            |mut tensor| tensor.eq_(rhs).to_kind(tch::Kind::Bool),
            |tensor| tensor.eq(rhs),
        );
        to_tensor(tensor)
    }

    fn bool_into_int<const D: usize>(
        tensor: TchTensor<bool, D>,
    ) -> <<TchBackend<E> as Backend>::IntegerBackend as Backend>::TensorPrimitive<D> {
        let tensor = tensor.tensor.to_kind(TchKind::<i64>::new().kind());
        to_tensor(tensor)
    }
}
