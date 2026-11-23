use burn_backend::ops::ReadFuture;
use burn_backend::{
    Backend,
    ops::{TransactionOps, TransactionPrimitive},
};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy};

impl<B: Backend, C: CheckpointStrategy> TransactionOps<Self> for Autodiff<B, C> {
    fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> impl ReadFuture<burn_backend::ops::TransactionPrimitiveData> {
        B::tr_execute(TransactionPrimitive::new(
            transaction
                .read_floats
                .into_iter()
                .map(|t| t.primitive)
                .collect(),
            transaction.read_qfloats,
            transaction.read_ints,
            transaction.read_bools,
        ))
    }
}
