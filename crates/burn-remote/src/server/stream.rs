use core::marker::PhantomData;
use std::sync::Arc;

use crate::shared::{ConnectionId, TaskResponse, TensorRemote};

use super::{
    processor::{Processor, ProcessorTask},
    tensor_data_service::TensorDataService,
};

use burn_ir::{BackendIr, OperationIr, TensorId, TensorIr};
use burn_router::Runner;
use burn_tensor::{DType, TensorData};
use tokio::sync::mpsc::{Receiver, Sender};

/// A stream makes sure all operations registered are executed in the order they were sent to the
/// server, protentially waiting to reconstruct consistency.
#[derive(Clone)]
pub struct Stream<B: BackendIr> {
    compute_sender: Sender<ProcessorTask>,
    writer_sender: Sender<Receiver<TaskResponse>>,
    _p: PhantomData<B>,
}

impl<B: BackendIr> Stream<B> {
    pub async fn new(
        runner: Runner<B>,
        writer_sender: Sender<Receiver<TaskResponse>>,
        state: Arc<TensorDataService>,
    ) -> Self {
        let sender = Processor::start(runner, state).await;

        Self {
            compute_sender: sender,
            writer_sender,
            _p: PhantomData,
        }
    }

    pub async fn register_operation(&self, op: Box<OperationIr>) {
        self.compute_sender
            .send(ProcessorTask::RegisterOperation(op))
            .await
            .unwrap();
    }

    pub async fn register_empty_tensor(
        &self,
        tensor_id: TensorId,
        shape: Vec<usize>,
        dtype: DType,
    ) {
        self.compute_sender
            .send(ProcessorTask::RegisterEmptyTensor(tensor_id, shape, dtype))
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

    pub async fn expose_tensor_remote(&self, tensor: TensorIr, count: u32) {
        self.compute_sender
            .send(ProcessorTask::ExposeTensorRemote { tensor, count })
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
}
