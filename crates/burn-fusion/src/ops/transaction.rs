use burn_tensor::{
    backend::ExecutionError,
    ops::{TransactionOps, TransactionPrimitive},
};

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> TransactionOps<Fusion<B>> for Fusion<B> {
    async fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> Result<burn_tensor::ops::TransactionPrimitiveData, ExecutionError> {
        B::tr_execute(TransactionPrimitive {
            read_floats: transaction
                .read_floats
                .into_iter()
                .map(|t| t.client.clone().resolve_tensor_float::<B>(t))
                .collect(),
            read_qfloats: transaction
                .read_qfloats
                .into_iter()
                .map(|_t| todo!("Quantization not supported yet"))
                .collect(),
            read_ints: transaction
                .read_ints
                .into_iter()
                .map(|t| t.client.clone().resolve_tensor_int::<B>(t))
                .collect(),
            read_bools: transaction
                .read_bools
                .into_iter()
                .map(|t| t.client.clone().resolve_tensor_bool::<B>(t))
                .collect(),
        })
        .await
    }
}
