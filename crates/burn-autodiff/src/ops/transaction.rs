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
            floats: transaction
                .floats
                .into_iter()
                .map(|t| t.primitive)
                .collect(),
            qfloats: transaction.qfloats,
            ints: transaction.ints,
            bools: transaction.bools,
        })
    }
}
