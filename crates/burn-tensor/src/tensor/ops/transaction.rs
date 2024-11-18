use core::future::Future;

use super::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use crate::{backend::Backend, TensorData};

pub struct Transaction<B: Backend> {
    floats: Vec<FloatTensor<B>>,
    qfloats: Vec<QuantizedTensor<B>>,
    ints: Vec<IntTensor<B>>,
    bools: Vec<BoolTensor<B>>,
}

pub struct TransactionResult {
    floats: Vec<TensorData>,
    qfloats: Vec<TensorData>,
    ints: Vec<TensorData>,
    bools: Vec<TensorData>,
}

pub trait TransactionOps<B: Backend> {
    fn execute(
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
