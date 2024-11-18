use burn_tensor::ops::{Transaction, TransactionOps};

use crate::{client::FusionClient, Fusion, FusionBackend};

impl<B: FusionBackend> TransactionOps<Fusion<B>> for Fusion<B> {
    fn tr_execute(
        transaction: Transaction<Self>,
    ) -> impl std::future::Future<Output = burn_tensor::ops::TransactionResult> + 'static + Send
    {
        B::tr_execute(Transaction {
            floats: transaction
                .floats
                .into_iter()
                .map(|t| t.client.clone().resolve_tensor_float::<B>(t))
                .collect(),
            qfloats: transaction
                .qfloats
                .into_iter()
                .map(|_t| todo!("Quantization not supported yet"))
                .collect(),
            ints: transaction
                .ints
                .into_iter()
                .map(|t| t.client.clone().resolve_tensor_int::<B>(t))
                .collect(),
            bools: transaction
                .bools
                .into_iter()
                .map(|t| t.client.clone().resolve_tensor_bool::<B>(t))
                .collect(),
        })
    }
}
