use core::future::Future;

use super::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use crate::{backend::Backend, TensorData};

#[derive(Default)]
pub struct Transaction<B: Backend> {
    pub floats: Vec<FloatTensor<B>>,
    pub qfloats: Vec<QuantizedTensor<B>>,
    pub ints: Vec<IntTensor<B>>,
    pub bools: Vec<BoolTensor<B>>,
}

#[derive(Default)]
pub struct TransactionResult {
    pub floats: Vec<TensorData>,
    pub qfloats: Vec<TensorData>,
    pub ints: Vec<TensorData>,
    pub bools: Vec<TensorData>,
}

pub trait TransactionOps<B: Backend> {
    fn tr_execute(
        transaction: Transaction<B>,
    ) -> impl Future<Output = TransactionResult> + 'static + Send {
        async move {
            let mut floats = Vec::new();
            let mut qfloats = Vec::new();
            let mut ints = Vec::new();
            let mut bools = Vec::new();

            for t in transaction.floats {
                floats.push(B::float_into_data(t).await);
            }
            for t in transaction.qfloats {
                qfloats.push(B::q_into_data(t).await);
            }
            for t in transaction.ints {
                ints.push(B::int_into_data(t).await);
            }
            for t in transaction.bools {
                bools.push(B::bool_into_data(t).await);
            }

            TransactionResult {
                floats,
                qfloats,
                ints,
                bools,
            }
        }
    }
}
