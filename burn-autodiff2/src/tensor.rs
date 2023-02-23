use burn_tensor::backend::Backend;

use crate::{graph::ops::OpsMapRef, ADBackendDecorator};

use burn_tensor::ops::*;

#[derive(Debug, Clone, Copy)]
pub enum Requirement {
    Grad,
    GradInBackward,
    None,
}

#[derive(Debug, Clone)]
pub struct ADTensor<B: Backend, const D: usize> {
    map: OpsMapRef<B>,
    primitive: B::TensorPrimitive<D>,
    usize: usize,
    requirement: Requirement,
}

pub type Elem<B> = <ADBackendDecorator<B> as Backend>::Elem;
pub type BoolTensor<B, const D: usize> = <ADBackendDecorator<B> as Backend>::BoolTensorPrimitive<D>;
pub type IntTensor<B, const D: usize> =
    <<ADBackendDecorator<B> as Backend>::IntegerBackend as Backend>::TensorPrimitive<D>;

impl<const D: usize, B> core::ops::Add<Self> for ADTensor<B, D>
where
    B: Backend,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        ADBackendDecorator::add(self, other)
    }
}

impl<B: Backend, const D: usize> Zeros for ADTensor<B, D> {
    fn zeros(&self) -> Self {
        todo!()
    }
}

impl<B: Backend, const D: usize> Ones for ADTensor<B, D> {
    fn ones(&self) -> Self {
        todo!()
    }
}
