use core::marker::PhantomData;
use std::sync::Arc;

use crate::shared::{ConnectionId, TaskResponse, TensorRemote};

use super::processor::{Processor, ProcessorTask};
use burn_backend::TensorData;
use burn_communication::{
    Protocol,
    data_service::{TensorDataService, TensorTransferId},
};
use burn_ir::{BackendIr, OperationIr, TensorId, TensorIr};
use burn_router::Runner;
use burn_std::DType;
use tokio::sync::mpsc::{Receiver, Sender};

/// A stream makes sure all operations registered are executed in the order they were sent to the
/// server, potentially waiting to reconstruct consistency.
#[derive(Clone)]
pub struct Stream<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    compute_sender: Sender<ProcessorTask>,
    writer_sender: Sender<Receiver<TaskResponse>>,
    _p: PhantomData<B>,
    _n: PhantomData<P>,
}

impl<B, P> Stream<B, P>
where
    B: BackendIr,
    P: Protocol,
{
    pub async fn new(
        runner: Runner<B>,
        writer_sender: Sender<Receiver<TaskResponse>>,
        data_service: Arc<TensorDataService<B, P>>,
    ) -> Self {
        let sender = Processor::<B, P>::start(runner, data_service).await;

        Self {
            compute_sender: sender,
            writer_sender,
            _p: PhantomData,
            _n: PhantomData,
        }
    }

    pub async fn register_operation(&self, op: Box<OperationIr>) {
        self.compute_sender
            .send(ProcessorTask::RegisterOperation(op))
            .await
            .unwrap();
    }

    pub async fn register_tensor(&self, tensor_id: TensorId, data: TensorData) {
        self.compute_sender
            .send(ProcessorTask::RegisterTensor(tensor_id, data))
            .await
            .unwrap();
    }

    pub async fn register_tensor_remote(&self, tensor: TensorRemote, new_id: TensorId) {
        self.compute_sender
            .send(ProcessorTask::RegisterTensorRemote(tensor, new_id))
            .await
            .unwrap();
    }

    pub async fn expose_tensor_remote(
        &self,
        tensor: TensorIr,
        count: u32,
        transfer_id: TensorTransferId,
    ) {
        self.compute_sender
            .send(ProcessorTask::ExposeTensorRemote {
                tensor,
                count,
                transfer_id,
            })
            .await
            .unwrap();
    }

    pub async fn read_tensor(&self, id: ConnectionId, desc: TensorIr) {
        let (callback_sender, callback_rec) = tokio::sync::mpsc::channel(1);

        self.compute_sender
            .send(ProcessorTask::ReadTensor(id, desc, callback_sender))
            .await
            .unwrap();

        self.writer_sender.send(callback_rec).await.unwrap();
    }

    pub async fn sync(&self, id: ConnectionId) {
        let (callback_sender, callback_rec) = tokio::sync::mpsc::channel(1);

        self.compute_sender
            .send(ProcessorTask::Sync(id, callback_sender))
            .await
            .unwrap();

        self.writer_sender.send(callback_rec).await.unwrap();
    }

    pub async fn close(&self) {
        self.compute_sender
            .send(ProcessorTask::Close)
            .await
            .unwrap();
    }

    pub async fn seed(&self, seed: u64) {
        self.compute_sender
            .send(ProcessorTask::Seed(seed))
            .await
            .unwrap();
    }

    pub async fn supports_dtype(&self, id: ConnectionId, dtype: DType) {
        let (callback_sender, callback_rec) = tokio::sync::mpsc::channel(1);

        self.compute_sender
            .send(ProcessorTask::SupportsDType(id, dtype, callback_sender))
            .await
            .unwrap();

        self.writer_sender.send(callback_rec).await.unwrap();
    }
}
