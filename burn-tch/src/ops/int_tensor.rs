use std::ops::Range;

use burn_tensor::{backend::Backend, ops::IntTensorOps, Data, Shape};

use crate::{element::TchElement, TchBackend, TchDevice, TchShape, TchTensor};

use super::TchOps;

impl<E: TchElement> IntTensorOps<TchBackend<E>> for TchBackend<E> {
    fn int_from_data<const D: usize>(data: Data<i64, D>, device: &TchDevice) -> TchTensor<i64, D> {
        TchTensor::from_data(data, (*device).into())
    }

    fn int_shape<const D: usize>(tensor: &TchTensor<i64, D>) -> Shape<D> {
        tensor.shape()
    }

    fn int_to_data<const D: usize>(tensor: &TchTensor<i64, D>) -> Data<i64, D> {
        let values: Vec<i64> = tensor.tensor.shallow_clone().into();
        Data::new(values, tensor.shape())
    }

    fn int_into_data<const D: usize>(tensor: TchTensor<i64, D>) -> Data<i64, D> {
        let shape = tensor.shape();
        let values: Vec<i64> = tensor.unary_ops(
            |tensor| tensor.into(),
            |tensor| tensor.shallow_clone().into(),
        );
        Data::new(values, shape)
    }

    fn int_to_device<const D: usize>(
        tensor: TchTensor<i64, D>,
        device: &TchDevice,
    ) -> TchTensor<i64, D> {
        TchTensor::new(tensor.tensor.to((*device).into()))
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: TchTensor<i64, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<i64, D2> {
        let shape_tch: TchShape<D2> = shape.into();
        let tensor = tensor.unary_ops(
            |mut tensor| tensor.resize_(&shape_tch.dims),
            |tensor| tensor.reshape(&shape_tch.dims),
        );

        TchTensor::new(tensor)
    }

    fn int_device<const D: usize>(tensor: &TchTensor<i64, D>) -> TchDevice {
        tensor.tensor.device().into()
    }

    fn int_empty<const D: usize>(
        shape: Shape<D>,
        device: &<TchBackend<E> as Backend>::Device,
    ) -> TchTensor<i64, D> {
        let tensor = tch::Tensor::empty(
            &shape.dims.map(|a| a as i64),
            (tch::Kind::Int64, (*device).into()),
        );

        TchTensor::new(tensor)
    }

    fn int_index<const D1: usize, const D2: usize>(
        tensor: TchTensor<i64, D1>,
        indexes: [Range<usize>; D2],
    ) -> TchTensor<i64, D1> {
        TchOps::index(tensor, indexes)
    }
    fn int_index_assign<const D1: usize, const D2: usize>(
        tensor: TchTensor<i64, D1>,
        indexes: [std::ops::Range<usize>; D2],
        value: TchTensor<i64, D1>,
    ) -> TchTensor<i64, D1> {
        TchOps::index_assign(tensor, indexes, value)
    }

    fn int_cat<const D: usize>(tensors: Vec<TchTensor<i64, D>>, dim: usize) -> TchTensor<i64, D> {
        TchOps::cat(tensors, dim)
    }

    fn int_equal<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::equal(lhs, rhs)
    }

    fn int_equal_elem<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<bool, D> {
        TchOps::equal_elem(lhs, rhs)
    }

    fn int_greater<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::greater(lhs, rhs)
    }

    fn int_greater_elem<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<bool, D> {
        TchOps::greater_elem(lhs, rhs)
    }

    fn int_greater_equal<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::greater_equal(lhs, rhs)
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: i64,
    ) -> TchTensor<bool, D> {
        TchOps::greater_equal_elem(lhs, rhs)
    }

    fn int_lower<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::lower(lhs, rhs)
    }

    fn int_lower_elem<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<bool, D> {
        TchOps::lower_elem(lhs, rhs)
    }

    fn int_lower_equal<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<bool, D> {
        TchOps::lower_equal(lhs, rhs)
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: i64,
    ) -> TchTensor<bool, D> {
        TchOps::lower_equal_elem(lhs, rhs)
    }

    fn int_add<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::add(lhs, rhs)
    }

    fn int_add_scalar<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<i64, D> {
        let tensor = lhs.unary_ops(
            |mut tensor| tensor.f_add_scalar_(rhs).unwrap(),
            |tensor| tensor.f_add_scalar(rhs).unwrap(),
        );
        TchTensor::new(tensor)
    }

    fn int_sub<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::sub(lhs, rhs)
    }

    fn int_sub_scalar<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<i64, D> {
        let tensor = lhs.unary_ops(
            |mut tensor| tensor.f_sub_scalar_(rhs).unwrap(),
            |tensor| tensor.f_sub_scalar(rhs).unwrap(),
        );
        TchTensor::new(tensor)
    }

    fn int_mul<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::mul(lhs, rhs)
    }

    fn int_mul_scalar<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<i64, D> {
        let tensor = lhs.unary_ops(
            |mut tensor| tensor.f_mul_scalar_(rhs).unwrap(),
            |tensor| tensor.f_mul_scalar(rhs).unwrap(),
        );
        TchTensor::new(tensor)
    }

    fn int_div<const D: usize>(
        lhs: TchTensor<i64, D>,
        rhs: TchTensor<i64, D>,
    ) -> TchTensor<i64, D> {
        TchOps::div(lhs, rhs)
    }

    fn int_div_scalar<const D: usize>(lhs: TchTensor<i64, D>, rhs: i64) -> TchTensor<i64, D> {
        let tensor = lhs.unary_ops(
            |mut tensor| tensor.f_div_scalar_(rhs).unwrap(),
            |tensor| tensor.f_div_scalar(rhs).unwrap(),
        );
        TchTensor::new(tensor)
    }

    fn int_neg<const D: usize>(tensor: TchTensor<i64, D>) -> TchTensor<i64, D> {
        Self::int_mul_scalar(tensor, -1)
    }

    fn int_zeros<const D: usize>(
        shape: Shape<D>,
        device: &<TchBackend<E> as Backend>::Device,
    ) -> TchTensor<i64, D> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::zeros(&shape.dims, (tch::Kind::Int64, device)))
    }

    fn int_ones<const D: usize>(
        shape: Shape<D>,
        device: &<TchBackend<E> as Backend>::Device,
    ) -> TchTensor<i64, D> {
        let shape = TchShape::from(shape);
        let device: tch::Device = (*device).into();

        TchTensor::new(tch::Tensor::ones(&shape.dims, (tch::Kind::Int64, device)))
    }

    fn int_sum<const D: usize>(tensor: TchTensor<i64, D>) -> TchTensor<i64, 1> {
        TchOps::sum(tensor)
    }

    fn int_sum_dim<const D: usize>(tensor: TchTensor<i64, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::sum_dim(tensor, dim)
    }

    fn int_mean<const D: usize>(tensor: TchTensor<i64, D>) -> TchTensor<i64, 1> {
        TchOps::mean(tensor)
    }

    fn int_mean_dim<const D: usize>(tensor: TchTensor<i64, D>, dim: usize) -> TchTensor<i64, D> {
        TchOps::mean_dim(tensor, dim)
    }
}
