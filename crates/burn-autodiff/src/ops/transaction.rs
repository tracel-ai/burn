use burn_tensor::{
    backend::Backend,
    ops::{Transaction, TransactionOps},
};

use crate::{checkpoint::strategy::CheckpointStrategy, Autodiff};

impl<B: Backend, C: CheckpointStrategy> TransactionOps<Self> for Autodiff<B, C> {
    fn tr_execute(
        transaction: Transaction<Self>,
    ) -> impl std::future::Future<Output = burn_tensor::ops::TransactionResult> + 'static + Send
    {
        B::tr_execute(Transaction {
            read_floats: transaction
                .read_floats
                .into_iter()
                .map(|t| t.primitive)
                .collect(),
            read_qfloats: transaction.read_qfloats,
            read_ints: transaction.read_ints,
            read_bools: transaction.read_bools,
        })
    }
}
