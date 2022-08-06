use super::{backend::Backend, Data, Distribution};
use crate::tensor::Shape;
use std::ops::Range;

pub trait TensorOpsUtilities<E, const D: usize> {
    fn shape(&self) -> &Shape<D>;
    fn into_data(self) -> Data<E, D>;
    fn to_data(&self) -> Data<E, D>;
}

pub trait TensorOpsDevice<B: Backend, const D: usize> {
    fn device(&self) -> B::Device;
    fn to_device(&self, device: B::Device) -> Self;
}

pub trait TensorCreationLike<E, const D: usize> {
    fn new_like_empty(&self) -> Self;
    fn new_like_random(&self, distribution: Distribution<E>) -> Self;
    fn new_like_data(&self, data: Data<E, D>) -> Self;
    fn new_like_zeros(&self) -> Self;
    fn new_like_ones(&self) -> Self;
}

pub trait TensorCreationFork<B: Backend, const D: usize> {
    fn new_fork_empty<const D2: usize>(&self, shape: Shape<D2>) -> B::TensorPrimitive<D2>;
    fn new_fork_random<const D2: usize>(
        &self,
        shape: Shape<D2>,
        distribution: Distribution<B::Elem>,
    ) -> B::TensorPrimitive<D2>;
    fn new_fork_data<const D2: usize>(&self, data: Data<B::Elem, D2>) -> B::TensorPrimitive<D2>;
    fn new_fork_zeros<const D2: usize>(&self, shape: Shape<D2>) -> B::TensorPrimitive<D2>;
    fn new_fork_ones<const D2: usize>(&self, shape: Shape<D2>) -> B::TensorPrimitive<D2>;
}

pub trait TensorOpsAdd<E, const D: usize>: std::ops::Add<Self, Output = Self>
where
    Self: Sized,
{
    fn add(&self, other: &Self) -> Self;
    fn add_scalar(&self, other: &E) -> Self;
}

pub trait TensorOpsSub<E, const D: usize> {
    fn sub(&self, other: &Self) -> Self;
    fn sub_scalar(&self, other: &E) -> Self;
}

pub trait TensorOpsTranspose<E, const D: usize> {
    fn transpose(&self) -> Self;
}

pub trait TensorOpsMatmul<E, const D: usize> {
    fn matmul(&self, other: &Self) -> Self;
}

pub trait TensorOpsNeg<E, const D: usize> {
    fn neg(&self) -> Self;
}

pub trait TensorOpsMul<E, const D: usize> {
    fn mul(&self, other: &Self) -> Self;
    fn mul_scalar(&self, other: &E) -> Self;
}

pub trait TensorOpsReshape<B: Backend, const D: usize> {
    fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> B::TensorPrimitive<D2>;
}

pub trait TensorOpsIndex<E, const D1: usize> {
    fn index<const D2: usize>(&self, indexes: [Range<usize>; D2]) -> Self;
    fn index_assign<const D2: usize>(&self, indexes: [Range<usize>; D2], values: &Self) -> Self;
}

pub trait TensorOpsMapComparison<B: Backend, const D: usize> {
    fn greater(&self, other: &Self) -> B::BoolTensorPrimitive<D>;
    fn greater_scalar(&self, other: &B::Elem) -> B::BoolTensorPrimitive<D>;
    fn greater_equal(&self, other: &Self) -> B::BoolTensorPrimitive<D>;
    fn greater_equal_scalar(&self, other: &B::Elem) -> B::BoolTensorPrimitive<D>;
    fn lower(&self, other: &Self) -> B::BoolTensorPrimitive<D>;
    fn lower_scalar(&self, other: &B::Elem) -> B::BoolTensorPrimitive<D>;
    fn lower_equal(&self, other: &Self) -> B::BoolTensorPrimitive<D>;
    fn lower_equal_scalar(&self, other: &B::Elem) -> B::BoolTensorPrimitive<D>;
}

pub trait ActivationOpsReLU<E, const D: usize> {
    fn relu(&self) -> Self;
}

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}

pub trait Ones<T> {
    fn ones(&self) -> T;
}
