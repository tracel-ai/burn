use super::Tensor;
use crate::tensor::backend::Backend;
use crate::tensor::ops::*;
use crate::tensor::{Data, Shape};

pub struct BoolTensor<B: Backend, const D: usize> {
    pub(crate) value: B::BoolTensorPrimitive<D>,
}

impl<B, const D: usize> BoolTensor<B, D>
where
    B: Backend,
{
    pub fn new(tensor: B::BoolTensorPrimitive<D>) -> Self {
        Self { value: tensor }
    }

    pub fn shape(&self) -> &Shape<D> {
        self.value.shape()
    }

    pub fn into_data(self) -> Data<bool, D> {
        self.value.into_data()
    }

    pub fn to_data(&self) -> Data<bool, D> {
        self.value.to_data()
    }

    pub fn from_data(data: Data<bool, D>) -> Self {
        let value = B::from_data_bool(data, B::Device::default());
        Self::new(value)
    }

    pub fn to_int(&self) -> Tensor<B::IntegerBackend, D> {
        Tensor::from_data(self.value.to_data().convert())
    }
}
