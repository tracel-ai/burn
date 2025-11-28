use super::{BasicOps, Tensor, TensorPrimitive};
use crate::{
    TensorData,
    backend::{Backend, DeferedError},
    ops::{BoolTensor, IntTensor, TransactionPrimitive},
};
use alloc::vec::Vec;

#[derive(Default)]
/// A transaction can [read](Self::register) multiple tensors at once with a single operation improving
/// compute utilization with optimized laziness.
///
/// # Example
///
/// ```rust,ignore
///  let [output_data, loss_data, targets_data] = Transaction::default()
///    .register(output)
///    .register(loss)
///    .register(targets)
///    .execute()
///    .try_into()
///    .expect("Correct amount of tensor data");
/// ```
pub struct Transaction<B: Backend> {
    op: TransactionPrimitive<B>,
    orders: Vec<Order>,
}

enum Order {
    Float(usize),
    QFloat(usize),
    Int(usize),
    Bool(usize),
}

impl<B: Backend> Transaction<B> {
    /// Add a [tensor](Tensor) to the transaction to be read.
    pub fn register<const D: usize, K: BasicOps<B>>(mut self, tensor: Tensor<B, D, K>) -> Self {
        K::register_transaction(&mut self, tensor.into_primitive());
        self
    }

    /// Executes the transaction synchronously and returns the [data](TensorData) in the same order
    /// in which they were [registered](Self::register).
    pub fn execute(self) -> Vec<TensorData> {
        burn_std::future::block_on(self.execute_async())
            .expect("Error while reading data: use `try_execute` to handle error at runtime")
    }

    /// Executes the transaction synchronously and returns the [data](TensorData) in the same order
    /// in which they were [registered](Self::register).
    pub fn try_execute(self) -> Result<Vec<TensorData>, DeferedError> {
        burn_std::future::block_on(self.execute_async())
    }

    /// Executes the transaction asynchronously and returns the [data](TensorData) in the same order
    /// in which they were [registered](Self::register).
    pub async fn execute_async(self) -> Result<Vec<TensorData>, DeferedError> {
        let result = B::tr_execute(self.op).await?;

        let mut floats: Vec<_> = result.read_floats.into_iter().map(Some).collect();
        let mut qfloats: Vec<_> = result.read_qfloats.into_iter().map(Some).collect();
        let mut ints: Vec<_> = result.read_ints.into_iter().map(Some).collect();
        let mut bools: Vec<_> = result.read_bools.into_iter().map(Some).collect();

        Ok(self
            .orders
            .into_iter()
            .map(|order| match order {
                Order::Float(index) => floats.get_mut(index).unwrap().take().unwrap(),
                Order::QFloat(index) => qfloats.get_mut(index).unwrap().take().unwrap(),
                Order::Int(index) => ints.get_mut(index).unwrap().take().unwrap(),
                Order::Bool(index) => bools.get_mut(index).unwrap().take().unwrap(),
            })
            .collect::<Vec<_>>())
    }

    pub(crate) fn register_float(&mut self, tensor: TensorPrimitive<B>) {
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

    pub(crate) fn register_int(&mut self, tensor: IntTensor<B>) {
        self.orders.push(Order::Int(self.op.read_ints.len()));
        self.op.read_ints.push(tensor);
    }

    pub(crate) fn register_bool(&mut self, tensor: BoolTensor<B>) {
        self.orders.push(Order::Bool(self.op.read_bools.len()));
        self.op.read_bools.push(tensor);
    }
}
