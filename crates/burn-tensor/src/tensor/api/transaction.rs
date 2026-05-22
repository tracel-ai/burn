use super::Tensor;
use crate::{ExecutionError, TensorData};
use alloc::vec::Vec;
use burn_backend::ops::TransactionPrimitive;
use burn_dispatch::Dispatch;

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
pub struct Transaction {
    opaque: transaction_opaque::Opaque,
}

burn_std::obfuscate!(
    type: TransactionPrimitive<Dispatch>,
    module: transaction_opaque,
    derives: [Send, Sync],
);

impl Default for Transaction {
    fn default() -> Self {
        Self::from_op(TransactionPrimitive::<Dispatch>::default())
    }
}

impl Transaction {
    /// Crate-internal constructor wrapping a dispatch-level transaction.
    pub(crate) fn from_op(op: TransactionPrimitive<Dispatch>) -> Self {
        Self {
            opaque: transaction_opaque::Opaque::new(op),
        }
    }

    /// Crate-internal mutable borrow of the underlying transaction primitive.
    pub(crate) fn as_op_mut(&mut self) -> &mut TransactionPrimitive<Dispatch> {
        self.opaque.as_mut()
    }

    /// Crate-internal owning extraction of the underlying transaction primitive.
    pub(crate) fn into_op(self) -> TransactionPrimitive<Dispatch> {
        self.opaque.into_inner()
    }

    /// Add a [tensor](Tensor) to the transaction to be read.
    pub fn register<const D: usize, K: crate::kind::Transaction>(
        mut self,
        tensor: Tensor<D, K>,
    ) -> Self {
        K::register_transaction(self.as_op_mut(), tensor.primitive);
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
        self.into_op().execute_async().await
    }
}
