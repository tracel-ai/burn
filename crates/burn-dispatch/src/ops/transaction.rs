use burn_backend::{
    ExecutionError,
    ops::{TransactionOps, TransactionPrimitive, TransactionPrimitiveData},
};

use crate::Dispatch;
use crate::backends::*;

impl TransactionOps<Self> for Dispatch {
    async fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> Result<TransactionPrimitiveData, ExecutionError> {
        let first_tensor = transaction
            .read_floats
            .first()
            .or(transaction.read_ints.first())
            .or(transaction.read_bools.first());

        match first_tensor {
            Some(tensor) => {
                transaction_op!(transaction, tensor)
            }
            None => Ok(TransactionPrimitiveData::default()),
        }
    }
}
