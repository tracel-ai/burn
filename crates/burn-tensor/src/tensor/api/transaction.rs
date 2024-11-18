use super::{Bool, Int, Tensor, TensorPrimitive};
use crate::{
    backend::Backend,
    ops::{BoolTensor, IntTensor},
    TensorData,
};

pub struct Transaction<B: Backend> {
    floats: Vec<TensorPrimitive<B>>,
    ints: Vec<IntTensor<B>>,
    bools: Vec<BoolTensor<B>>,
}

pub struct TransactionResult {
    pub floats: Vec<TensorData>,
    pub ints: Vec<TensorData>,
    pub bools: Vec<TensorData>,
}

impl<B: Backend> Transaction<B> {
    pub fn float<const D: usize>(mut self, tensor: Tensor<B, D>) -> Self {
        self.floats.push(tensor.into_primitive());
        self
    }
    pub fn int<const D: usize>(mut self, tensor: Tensor<B, D, Int>) -> Self {
        self.ints.push(tensor.into_primitive());
        self
    }
    pub fn bool<const D: usize>(mut self, tensor: Tensor<B, D, Bool>) -> Self {
        self.bools.push(tensor.into_primitive());
        self
    }

    pub fn execute(self) -> TransactionResult {
        todo!()
    }
}
