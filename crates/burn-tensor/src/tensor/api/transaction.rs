use super::{Bool, Int, Tensor, TensorPrimitive};
use crate::{backend::Backend, ops::Transaction, TensorData};

#[derive(Default)]
pub struct TransactionBuilder<B: Backend> {
    op: Transaction<B>,
    orders: Vec<Order>,
}

enum Order {
    Float(usize),
    QFloat(usize),
    Int(usize),
    Bool(usize),
}

impl<B: Backend> TransactionBuilder<B> {
    pub fn float<const D: usize>(mut self, tensor: Tensor<B, D>) -> Self {
        match tensor.into_primitive() {
            TensorPrimitive::Float(tensor) => {
                self.orders.push(Order::Float(self.op.floats.len()));
                self.op.floats.push(tensor);
            }
            TensorPrimitive::QFloat(tensor) => {
                self.orders.push(Order::QFloat(self.op.qfloats.len()));
                self.op.qfloats.push(tensor);
            }
        }
        self
    }
    pub fn int<const D: usize>(mut self, tensor: Tensor<B, D, Int>) -> Self {
        self.orders.push(Order::Int(self.op.ints.len()));
        self.op.ints.push(tensor.into_primitive());
        self
    }
    pub fn bool<const D: usize>(mut self, tensor: Tensor<B, D, Bool>) -> Self {
        self.orders.push(Order::Bool(self.op.bools.len()));
        self.op.bools.push(tensor.into_primitive());
        self
    }

    pub fn execute(self) -> Vec<TensorData> {
        let result = burn_common::future::block_on(B::tr_execute(self.op));

        let mut floats: Vec<_> = result.floats.into_iter().map(|t| Some(t)).collect();
        let mut qfloats: Vec<_> = result.qfloats.into_iter().map(|t| Some(t)).collect();
        let mut ints: Vec<_> = result.ints.into_iter().map(|t| Some(t)).collect();
        let mut bools: Vec<_> = result.bools.into_iter().map(|t| Some(t)).collect();

        self.orders
            .into_iter()
            .map(|order| match order {
                Order::Float(index) => floats.get_mut(index).unwrap().take().unwrap(),
                Order::QFloat(index) => qfloats.get_mut(index).unwrap().take().unwrap(),
                Order::Int(index) => ints.get_mut(index).unwrap().take().unwrap(),
                Order::Bool(index) => bools.get_mut(index).unwrap().take().unwrap(),
            })
            .collect::<Vec<_>>()
    }
}
