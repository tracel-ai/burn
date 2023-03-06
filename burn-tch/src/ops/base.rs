use crate::{TchShape, TchTensor};
use std::{marker::PhantomData, ops::Range};

pub struct TchOps<E: tch::kind::Element + Copy + Default> {
    e: PhantomData<E>,
}

impl<E: tch::kind::Element + Copy + Default> TchOps<E> {
    pub fn index<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        indexes: [Range<usize>; D2],
    ) -> TchTensor<E, D1> {
        let mut tensor = tensor.tensor.shallow_clone();

        for (i, index) in indexes.iter().enumerate().take(D2) {
            let start = index.start as i64;
            let length = (index.end - index.start) as i64;
            tensor = tensor.narrow(i as i64, start, length);
        }

        TchTensor::new(tensor)
    }

    pub fn index_assign<const D1: usize, const D2: usize>(
        tensor: TchTensor<E, D1>,
        indexes: [Range<usize>; D2],
        value: TchTensor<E, D1>,
    ) -> TchTensor<E, D1> {
        let tensor_original = tensor.tensor.copy();
        let tch_shape = TchShape::from(tensor.shape());

        let mut tensor = tensor_original.view_(&tch_shape.dims);

        for (i, index) in indexes.into_iter().enumerate().take(D2) {
            let start = index.start as i64;
            let length = (index.end - index.start) as i64;

            tensor = tensor.narrow(i as i64, start, length);
        }

        tensor.copy_(&value.tensor);

        TchTensor::new(tensor_original)
    }

    pub fn cat<const D: usize>(tensors: Vec<TchTensor<E, D>>, dim: usize) -> TchTensor<E, D> {
        let tensors: Vec<tch::Tensor> = tensors
            .into_iter()
            .map(|t| t.tensor.shallow_clone())
            .collect();
        let tensor = tch::Tensor::cat(&tensors, dim as i64);

        TchTensor::new(tensor)
    }

    pub fn equal<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<bool, D> {
        let tensor = TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.eq_tensor_(rhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| rhs.eq_tensor_(lhs).to_kind(tch::Kind::Bool),
            |lhs, rhs| lhs.eq_tensor(rhs),
        );

        TchTensor::new(tensor)
    }

    pub fn add<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_add_(rhs).unwrap(),
            |lhs, rhs| rhs.f_add_(lhs).unwrap(),
            |lhs, rhs| lhs.f_add(rhs).unwrap(),
        );

        TchTensor::new(tensor)
    }

    pub fn sub<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_sub_(rhs).unwrap(),
            |lhs, rhs| lhs.f_sub(rhs).unwrap(),
            |lhs, rhs| lhs.f_sub(rhs).unwrap(),
        );

        TchTensor::new(tensor)
    }

    pub fn mul<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_mul_(rhs).unwrap(),
            |lhs, rhs| rhs.f_mul_(lhs).unwrap(),
            |lhs, rhs| lhs.f_mul(rhs).unwrap(),
        );
        TchTensor::new(tensor)
    }

    pub fn div<const D: usize>(lhs: TchTensor<E, D>, rhs: TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = TchTensor::binary_ops_tensor(
            lhs,
            rhs,
            |lhs, rhs| lhs.f_div_(rhs).unwrap(),
            |lhs, rhs| lhs.f_div(rhs).unwrap(),
            |lhs, rhs| lhs.f_div(rhs).unwrap(),
        );

        TchTensor::new(tensor)
    }

    pub fn mean<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        let tensor = tensor.tensor.mean(E::KIND);
        TchTensor::new(tensor)
    }

    pub fn sum<const D: usize>(tensor: TchTensor<E, D>) -> TchTensor<E, 1> {
        let tensor = tensor.tensor.sum(E::KIND);
        TchTensor::new(tensor)
    }

    pub fn mean_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        let tensor = tensor
            .tensor
            .mean_dim(Some([dim as i64].as_slice()), true, E::KIND);
        TchTensor::new(tensor)
    }

    pub fn sum_dim<const D: usize>(tensor: TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        let tensor = tensor
            .tensor
            .sum_dim_intlist(Some([dim as i64].as_slice()), true, E::KIND);
        TchTensor::new(tensor)
    }
}
