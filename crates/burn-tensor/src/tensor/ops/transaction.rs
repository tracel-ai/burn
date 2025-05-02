use alloc::vec::Vec;
use core::future::Future;

use super::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use crate::{TensorData, backend::Backend};

#[derive(Default)]
/// Contains all tensor primitives that are going to be read.
pub struct TransactionPrimitive<B: Backend> {
    /// Float tensors.
    pub read_floats: Vec<FloatTensor<B>>,
    /// Quantized tensors.
    pub read_qfloats: Vec<QuantizedTensor<B>>,
    /// Int tensors.
    pub read_ints: Vec<IntTensor<B>>,
    /// Bool tensors.
    pub read_bools: Vec<BoolTensor<B>>,
}

#[derive(Default)]
/// Contains all [data](TensorData) related to a [transaction](TransactionPrimitive).
pub struct TransactionPrimitiveResult {
    /// Float tensor data.
    pub read_floats: Vec<TensorData>,
    /// Quantized tensor data.
    pub read_qfloats: Vec<TensorData>,
    /// Int tensor data.
    pub read_ints: Vec<TensorData>,
    /// Bool tensor data.
    pub read_bools: Vec<TensorData>,
}

/// Operations that are sync by nature and that can be batch together in transactions to improve
/// compute utilization with efficient laziness.
pub trait TransactionOps<B: Backend> {
    /// Executes a [transaction](TransactionPrimitive) and return its
    /// [result](TransactionPrimitiveResult).
    fn tr_execute(
        transaction: TransactionPrimitive<B>,
    ) -> impl Future<Output = TransactionPrimitiveResult> + Send {
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

            TransactionPrimitiveResult {
                read_floats: floats,
                read_qfloats: qfloats,
                read_ints: ints,
                read_bools: bools,
            }
        }
    }
}
