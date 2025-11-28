use burn_tensor::{
    backend::{Backend, ExecutionError},
    ops::{TransactionOps, TransactionPrimitive},
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: Backend, C: CheckpointStrategy> TransactionOps<Self> for Autodiff<B, C> {
    async fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> Result<burn_tensor::ops::TransactionPrimitiveData, ExecutionError> {
        B::tr_execute(TransactionPrimitive {
            read_floats: transaction
                .read_floats
                .into_iter()
                .map(|t| t.primitive)
                .collect(),
            read_qfloats: transaction.read_qfloats,
            read_ints: transaction.read_ints,
            read_bools: transaction.read_bools,
        })
        .await
    }
}
