use super::{Backend, Data, Distribution, Element, Tensor, TensorType};
use crate::tensor::Shape;
use std::ops::Range;

pub trait TensorOpsDevice<E, const D: usize, B: Backend<E = E>> {
    fn device(&self) -> B::Device;
    fn to_device(&self, device: B::Device) -> Self;
}

pub type Allo<E, const D: usize> = Box<dyn TensorOpsUtilities<E, D>>;

pub trait TensorOpsUtilities<E, const D: usize> {
    fn shape(&self) -> &Shape<D>;
    fn into_data(self) -> Data<E, D>;
    fn to_data(&self) -> Data<E, D>;
}

pub trait TensorOpsCreation<E, const D: usize, B> {
    fn grad(&self) -> Tensor<D, B>
    where
        B: Backend<E = E> + TensorType<D, B>;
}

pub trait TensorCreationLike<E, const D: usize> {
    fn new_like_empty(&self) -> Self;
    fn new_like_random(&self, distribution: Distribution<E>) -> Self;
    fn new_like_data(&self, data: Data<E, D>) -> Self;
    fn new_like_zeros(&self) -> Self;
    fn new_like_ones(&self) -> Self;
}

pub trait TensorCreationFork<E, const D: usize, const D2: usize> {
    type Output;
    fn new_fork_empty(&self, shape: Shape<D2>) -> Self::Output;
    fn new_fork_random(&self, shape: Shape<D2>, distribution: Distribution<E>) -> Self::Output;
    fn new_fork_data(&self, data: Data<E, D2>) -> Self::Output;
    fn new_fork_zeros(&self, shape: Shape<D2>) -> Self::Output;
    fn new_fork_ones(&self, shape: Shape<D2>) -> Self::Output;
}

pub trait TensorOpsAdd<E, const D: usize>:
    std::ops::Add<Self, Output = Self> + std::ops::Add<E, Output = Self>
where
    Self: Sized,
{
    fn add(&self, other: &Self) -> Self;
    fn add_scalar(&self, other: &E) -> Self;
}

pub trait TensorOpsSub<E, const D: usize>:
    std::ops::Sub<Self, Output = Self> + std::ops::Sub<E, Output = Self>
where
    Self: Sized,
{
    fn sub(&self, other: &Self) -> Self;
    fn sub_scalar(&self, other: &E) -> Self;
}

pub trait TensorOpsTranspose<E, const D: usize> {
    fn transpose(&self) -> Self;
}

pub trait TensorOpsMatmul<E, const D: usize> {
    fn matmul(&self, other: &Self) -> Self;
}

pub trait TensorOpsNeg<E, const D: usize>: std::ops::Neg<Output = Self> {
    fn neg(&self) -> Self;
}

pub trait TensorOpsMul<E, const D: usize>:
    std::ops::Mul<E, Output = Self> + std::ops::Mul<Self, Output = Self>
where
    Self: Sized,
{
    fn mul(&self, other: &Self) -> Self;
    fn mul_scalar(&self, other: &E) -> Self;
}

pub trait TensorOpsReshape<E: Element, const D1: usize, const D2: usize, T> {
    fn reshape(&self, shape: Shape<D2>) -> T;
}

pub trait TensorOpsIndex<E, const D1: usize, const D2: usize> {
    fn index(&self, indexes: [Range<usize>; D2]) -> Self;
    fn index_assign(&self, indexes: [Range<usize>; D2], values: &Self) -> Self;
}

pub trait Zeros<T> {
    fn zeros(&self) -> T;
}

pub trait Ones<T> {
    fn ones(&self) -> T;
}
