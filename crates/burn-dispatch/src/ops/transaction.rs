use alloc::vec::Vec;
use burn_backend::ops::ReadFuture;
use burn_backend::ops::{TransactionOps, TransactionPrimitive, TransactionPrimitiveData};

use crate::Dispatch;

impl TransactionOps<Self> for Dispatch {
    fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> impl ReadFuture<TransactionPrimitiveData> {
        let first_tensor = transaction
            .read_floats
            .first()
            .or(transaction.read_ints.first())
            .or(transaction.read_bools.first());

        match first_tensor {
            Some(tensor) => {
                transaction_op!(transaction, tensor)
            }
            None => {
                let read: crate::backend::BoxedRead<TransactionPrimitiveData> =
                    alloc::boxed::Box::pin(async { Ok(TransactionPrimitiveData::default()) });
                read
            }
        }
    }
}
