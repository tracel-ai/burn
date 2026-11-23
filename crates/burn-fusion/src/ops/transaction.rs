use burn_backend::ops::ReadFuture;
use burn_backend::{
    backend::ExecutionError,
    ops::{TransactionOps, TransactionPrimitive, TransactionPrimitiveData},
};

use burn_std::future::DynFut;

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> TransactionOps<Fusion<B>> for Fusion<B> {
    fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> impl ReadFuture<TransactionPrimitiveData> {
        let read: DynFut<Result<TransactionPrimitiveData, ExecutionError>> =
            match resolve::<B>(transaction) {
                Ok(transaction) => Box::pin(B::tr_execute(transaction)),
                Err(err) => Box::pin(async move { Err(err) }),
            };
        read
    }
}

/// The transaction's tensors resolved to `B`'s, once fusion has run what writes them.
fn resolve<B: FusionBackend>(
    transaction: TransactionPrimitive<Fusion<B>>,
) -> Result<TransactionPrimitive<B>, ExecutionError> {
    if !transaction.read_qfloats.is_empty() {
        return Err(ExecutionError::generic(
            "A fusion transaction cannot read quantized tensors yet",
        ));
    }
    let floats = transaction
        .read_floats
        .into_iter()
        .map(|t| t.client.clone().resolve_tensor_float::<B>(t))
        .collect::<Result<_, _>>()?;
    let ints = transaction
        .read_ints
        .into_iter()
        .map(|t| t.client.clone().resolve_tensor_int::<B>(t))
        .collect::<Result<_, _>>()?;
    let bools = transaction
        .read_bools
        .into_iter()
        .map(|t| t.client.clone().resolve_tensor_bool::<B>(t))
        .collect::<Result<_, _>>()?;

    Ok(TransactionPrimitive::new(floats, Vec::new(), ints, bools))
}
