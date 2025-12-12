use burn_backend::{
    backend::ExecutionError,
    ops::{TransactionOps, TransactionPrimitive},
};

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> TransactionOps<Fusion<B>> for Fusion<B> {
    async fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> Result<burn_backend::ops::TransactionPrimitiveData, ExecutionError> {
        B::tr_execute(TransactionPrimitive::new(
            transaction
                .read_floats
                .into_iter()
                .map(|t| t.client.clone().resolve_tensor_float::<B>(t))
                .collect(),
            transaction
                .read_qfloats
                .into_iter()
                .map(|_t| todo!("Quantization not supported yet"))
                .collect(),
            transaction
                .read_ints
                .into_iter()
                .map(|t| t.client.clone().resolve_tensor_int::<B>(t))
                .collect(),
            transaction
                .read_bools
                .into_iter()
                .map(|t| t.client.clone().resolve_tensor_bool::<B>(t))
                .collect(),
        ))
        .await
    }
}
