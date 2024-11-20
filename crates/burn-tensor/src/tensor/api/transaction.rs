use super::{BasicOps, Tensor, TensorPrimitive};
use crate::{
    backend::Backend,
    ops::{BoolTensor, IntTensor, Transaction},
    TensorData,
};

#[derive(Default)]
pub struct TransactionQuery<B: Backend> {
    op: Transaction<B>,
    orders: Vec<Order>,
}

enum Order {
    Float(usize),
    QFloat(usize),
    Int(usize),
    Bool(usize),
}

impl<B: Backend> TransactionQuery<B> {
    pub fn read<const D: usize, K: BasicOps<B>>(mut self, tensor: Tensor<B, D, K>) -> Self {
        K::read(&mut self, tensor.into_primitive());
        self
    }

    pub(crate) fn read_float(&mut self, tensor: TensorPrimitive<B>) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                self.orders.push(Order::Float(self.op.read_floats.len()));
                self.op.read_floats.push(tensor);
            }
            TensorPrimitive::QFloat(tensor) => {
                self.orders.push(Order::QFloat(self.op.read_qfloats.len()));
                self.op.read_qfloats.push(tensor);
            }
        }
    }

    pub(crate) fn read_int(&mut self, tensor: IntTensor<B>) {
        self.orders.push(Order::Int(self.op.read_ints.len()));
        self.op.read_ints.push(tensor);
    }

    pub(crate) fn read_bool(&mut self, tensor: BoolTensor<B>) {
        self.orders.push(Order::Bool(self.op.read_bools.len()));
        self.op.read_bools.push(tensor);
    }

    pub fn execute(self) -> Vec<TensorData> {
        let result = burn_common::future::block_on(B::tr_execute(self.op));

        let mut floats: Vec<_> = result.read_floats.into_iter().map(|t| Some(t)).collect();
        let mut qfloats: Vec<_> = result.read_qfloats.into_iter().map(|t| Some(t)).collect();
        let mut ints: Vec<_> = result.read_ints.into_iter().map(|t| Some(t)).collect();
        let mut bools: Vec<_> = result.read_bools.into_iter().map(|t| Some(t)).collect();

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
