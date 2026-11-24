use super::ReadFuture;
use alloc::vec::Vec;

use crate::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use crate::{Backend, ExecutionError, TensorData, TensorPrimitive};

enum Order {
    Float(usize),
    QFloat(usize),
    Int(usize),
    Bool(usize),
}

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
    orders: Vec<Order>,
}

#[derive(Default)]
/// Contains all [data](TensorData) related to a [transaction](TransactionPrimitive).
pub struct TransactionPrimitiveData {
    /// Float tensor data.
    pub read_floats: Vec<TensorData>,
    /// Quantized tensor data.
    pub read_qfloats: Vec<TensorData>,
    /// Int tensor data.
    pub read_ints: Vec<TensorData>,
    /// Bool tensor data.
    pub read_bools: Vec<TensorData>,
}

async fn join_reads<F>(reads: Vec<F>) -> Result<Vec<TensorData>, ExecutionError>
where
    F: ReadFuture<TensorData>,
{
    let mut data = Vec::with_capacity(reads.len());
    for read in reads {
        data.push(read.await?);
    }
    Ok(data)
}

/// Operations that are sync by nature and that can be batch together in transactions to improve
/// compute utilization with efficient laziness.
pub trait TransactionOps<B: Backend> {
    /// Executes a [transaction](TransactionPrimitive) and return its
    /// [data](TransactionPrimitiveData).
    ///
    /// Reads the tensors as of this call. The caller may queue more work on the same stream
    /// before awaiting the result.
    fn tr_execute(
        transaction: TransactionPrimitive<B>,
    ) -> impl ReadFuture<TransactionPrimitiveData> {
        let floats: Vec<_> = transaction
            .read_floats
            .into_iter()
            .map(B::float_into_data)
            .collect();
        let qfloats: Vec<_> = transaction
            .read_qfloats
            .into_iter()
            .map(B::q_into_data)
            .collect();
        let ints: Vec<_> = transaction
            .read_ints
            .into_iter()
            .map(B::int_into_data)
            .collect();
        let bools: Vec<_> = transaction
            .read_bools
            .into_iter()
            .map(B::bool_into_data)
            .collect();

        async move {
            let floats = join_reads(floats).await?;
            let qfloats = join_reads(qfloats).await?;
            let ints = join_reads(ints).await?;
            let bools = join_reads(bools).await?;

            Ok(TransactionPrimitiveData {
                read_floats: floats,
                read_qfloats: qfloats,
                read_ints: ints,
                read_bools: bools,
            })
        }
    }
}

impl<B: Backend> TransactionPrimitive<B> {
    /// Creates a new transaction.
    pub fn new(
        read_floats: Vec<FloatTensor<B>>,
        read_qfloats: Vec<QuantizedTensor<B>>,
        read_ints: Vec<IntTensor<B>>,
        read_bools: Vec<BoolTensor<B>>,
    ) -> Self {
        Self {
            read_floats,
            read_qfloats,
            read_ints,
            read_bools,
            orders: Vec::default(),
        }
    }
    /// Executes the transaction asynchronously and returns the [data](TensorData) in the same order
    /// in which they were registered.
    pub fn execute_async(mut self) -> impl ReadFuture<Vec<TensorData>> {
        let mut orders = Vec::new();
        core::mem::swap(&mut orders, &mut self.orders);
        let read = B::tr_execute(self);
        async move { Ok(Self::in_order(orders, read.await?)) }
    }

    fn in_order(orders: Vec<Order>, result: TransactionPrimitiveData) -> Vec<TensorData> {
        let mut floats: Vec<_> = result.read_floats.into_iter().map(Some).collect();
        let mut qfloats: Vec<_> = result.read_qfloats.into_iter().map(Some).collect();
        let mut ints: Vec<_> = result.read_ints.into_iter().map(Some).collect();
        let mut bools: Vec<_> = result.read_bools.into_iter().map(Some).collect();

        orders
            .into_iter()
            .map(|order| match order {
                Order::Float(index) => floats.get_mut(index).unwrap().take().unwrap(),
                Order::QFloat(index) => qfloats.get_mut(index).unwrap().take().unwrap(),
                Order::Int(index) => ints.get_mut(index).unwrap().take().unwrap(),
                Order::Bool(index) => bools.get_mut(index).unwrap().take().unwrap(),
            })
            .collect::<Vec<_>>()
    }

    /// Register a float (or quantized-float) tensor to read in this transaction.
    pub fn register_float(&mut self, tensor: TensorPrimitive<B>) {
        match tensor {
            TensorPrimitive::Float(tensor) => {
                self.orders.push(Order::Float(self.read_floats.len()));
                self.read_floats.push(tensor);
            }
            TensorPrimitive::QFloat(tensor) => {
                self.orders.push(Order::QFloat(self.read_qfloats.len()));
                self.read_qfloats.push(tensor);
            }
        }
    }

    /// Register an int tensor to read in this transaction.
    pub fn register_int(&mut self, tensor: IntTensor<B>) {
        self.orders.push(Order::Int(self.read_ints.len()));
        self.read_ints.push(tensor);
    }

    /// Register a bool tensor to read in this transaction.
    pub fn register_bool(&mut self, tensor: BoolTensor<B>) {
        self.orders.push(Order::Bool(self.read_bools.len()));
        self.read_bools.push(tensor);
    }
}
