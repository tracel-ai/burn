use super::{BasicOps, Tensor};
use crate::{
    TensorData,
    backend::{Backend, ExecutionError},
    ops::TransactionPrimitive,
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
}

impl<B: Backend> Transaction<B> {
    /// Add a [tensor](Tensor) to the transaction to be read.
    pub fn register<const D: usize, K: BasicOps<B>>(mut self, tensor: Tensor<B, D, K>) -> Self {
        K::register_transaction(&mut self.op, tensor.into_primitive());
        self
    }

    /// Executes the transaction synchronously and returns the [data](TensorData) in the same order
    /// in which they were [registered](Self::register).
    pub fn execute(self) -> Vec<TensorData> {
        burn_std::future::block_on(self.execute_async())
            .expect("Error while reading data: use `try_execute` to handle error at runtime")
    }

    /// Executes the transaction synchronously and returns the [data](TensorData) in the same
    /// order in which they were [registered](Self::register).
    ///
    /// # Returns
    ///
    /// Any error that might have occurred since the last time the device was synchronized.
    pub fn try_execute(self) -> Result<Vec<TensorData>, ExecutionError> {
        burn_std::future::block_on(self.execute_async())
    }

    /// Executes the transaction asynchronously and returns the [data](TensorData) in the same order
    /// in which they were [registered](Self::register).
    pub async fn execute_async(self) -> Result<Vec<TensorData>, ExecutionError> {
        self.op.execute_async().await
    }
}
