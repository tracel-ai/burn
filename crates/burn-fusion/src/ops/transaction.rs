use burn_tensor::ops::{TransactionOps, TransactionPrimitive};

use crate::{Fusion, FusionBackend, client::FusionClient};

impl<B: FusionBackend> TransactionOps<Fusion<B>> for Fusion<B> {
    fn tr_execute(
        transaction: TransactionPrimitive<Self>,
    ) -> impl std::future::Future<Output = burn_tensor::ops::TransactionPrimitiveResult> + 'static + Send
    {
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
    }
}
