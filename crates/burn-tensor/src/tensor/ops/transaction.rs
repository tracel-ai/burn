use core::future::Future;

use super::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use crate::{backend::Backend, TensorData};

#[derive(Default)]
pub struct Transaction<B: Backend> {
    pub read_floats: Vec<FloatTensor<B>>,
    pub read_qfloats: Vec<QuantizedTensor<B>>,
    pub read_ints: Vec<IntTensor<B>>,
    pub read_bools: Vec<BoolTensor<B>>,
}

#[derive(Default)]
pub struct TransactionResult {
    pub read_floats: Vec<TensorData>,
    pub read_qfloats: Vec<TensorData>,
    pub read_ints: Vec<TensorData>,
    pub read_bools: Vec<TensorData>,
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

            for t in transaction.read_floats {
                floats.push(B::float_into_data(t).await);
            }
            for t in transaction.read_qfloats {
                qfloats.push(B::q_into_data(t).await);
            }
            for t in transaction.read_ints {
                ints.push(B::int_into_data(t).await);
            }
            for t in transaction.read_bools {
                bools.push(B::bool_into_data(t).await);
            }

            TransactionResult {
                read_floats: floats,
                read_qfloats: qfloats,
                read_ints: ints,
                read_bools: bools,
            }
        }
    }
}
