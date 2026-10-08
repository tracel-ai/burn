use burn_backend::{
    backend::ExecutionError,
    ops::{TransactionOps, TransactionPrimitive, TransactionPrimitiveData},
};

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> TransactionOps<Fusion<B>> for Fusion<B> {
    async fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> Result<TransactionPrimitiveData, ExecutionError> {
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

        B::tr_execute(TransactionPrimitive::new(floats, Vec::new(), ints, bools)).await
    }
}
