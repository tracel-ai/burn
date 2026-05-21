use super::Tensor;
use crate::{ExecutionError, TensorData};
use alloc::vec::Vec;
use burn_backend::ops::TransactionPrimitive;
use burn_dispatch::Dispatch;
use core::mem::MaybeUninit;

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
    blob: TransactionBlob,
}

type TransactionInner = MaybeUninit<TransactionPrimitive<Dispatch>>;

/// Storage for [`Transaction`]. Holds the raw bytes of a
/// [`TransactionPrimitive<Dispatch>`].
///
/// Intentionally has no type-level alignment marker (e.g.
/// `[TransactionInner; 0]`), since that would re-introduce a `burn_dispatch`
/// dependency in the type itself and undermine the compile-time goal of this
/// obfuscation. Alignment must therefore be handled at access sites
/// (TODO: bring back proper alignment without leaking the type).
struct TransactionBlob {
    bytes: [u8; size_of::<TransactionInner>()],
}

impl Drop for Transaction {
    fn drop(&mut self) {
        unsafe {
            let inner: &mut TransactionInner =
                &mut *(self.blob.bytes.as_mut_ptr() as *mut TransactionInner);
            inner.assume_init_drop();
        }
    }
}

impl Default for Transaction {
    fn default() -> Self {
        Self::from_op(TransactionPrimitive::<Dispatch>::default())
    }
}

impl Transaction {
    /// Crate-internal constructor wrapping a dispatch-level transaction.
    pub(crate) fn from_op(op: TransactionPrimitive<Dispatch>) -> Self {
        let mut blob = TransactionBlob {
            bytes: [0u8; size_of::<TransactionInner>()],
        };
        unsafe {
            let dst = blob.bytes.as_mut_ptr() as *mut TransactionInner;
            dst.write(MaybeUninit::new(op));
        }
        Self { blob }
    }

    /// Crate-internal mutable borrow of the underlying transaction primitive.
    pub(crate) fn as_op_mut(&mut self) -> &mut TransactionPrimitive<Dispatch> {
        unsafe {
            let inner: &mut TransactionInner =
                &mut *(self.blob.bytes.as_mut_ptr() as *mut TransactionInner);
            inner.assume_init_mut()
        }
    }

    /// Crate-internal owning extraction of the underlying transaction primitive.
    pub(crate) fn into_op(self) -> TransactionPrimitive<Dispatch> {
        unsafe {
            let inner: TransactionInner =
                core::ptr::read(self.blob.bytes.as_ptr() as *const TransactionInner);
            core::mem::forget(self);
            inner.assume_init()
        }
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
