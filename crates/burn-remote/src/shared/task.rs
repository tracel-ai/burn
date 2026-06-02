use burn_backend::{DTypeUsageSet, ExecutionError, TensorData};
use burn_communication::{Address, data_service::TensorTransferId};
use burn_ir::{OperationIr, TensorId, TensorIr};
use burn_std::{
    DType, DeviceSettings,
    id::{IdGenerator, StreamId},
};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// Routing id for a request that expects a response.
///
/// Only the response-producing compute tasks ([`ComputeTask::ReadTensor`],
/// [`ComputeTask::SyncBackend`], [`ComputeTask::DTypeUsage`]) carry a `RequestId`; the
/// server echoes it on its [`TaskResponse`] so the client demultiplexes responses back
/// to the right pending callback. Fire-and-forget tasks have no id because no response
/// ever comes back.
pub type RequestId = u64;

/// Unique identifier that can represent a session.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct SessionId {
    id: u64,
}

impl Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SessionId({})", self.id)
    }
}

impl SessionId {
    /// Create a new [session id](SessionId).
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            id: IdGenerator::generate(),
        }
    }
}

#[allow(missing_docs, clippy::large_enum_variant)]
#[derive(Serialize, Deserialize, Debug)]
pub enum Task {
    Compute(ComputeTask),
    /// Open a session bound to the device at the given index on the server.
    Init(SessionId, u32),
    Close(SessionId),
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TensorRemote {
    pub transfer_id: TensorTransferId,
    pub address: Address,
}

#[allow(missing_docs, clippy::large_enum_variant)]
#[derive(Serialize, Deserialize, Debug)]
pub enum ComputeTask {
    Seed(u64),
    /// A single [`OperationIr`] tagged with the stream it was issued on.
    ///
    /// Each op is sent independently — batching across streams would lose stream
    /// identity, and the server-side backend (fusion, etc.) handles any batching of
    /// its own.
    RegisterOperation(StreamId, OperationIr),
    RegisterTensor(StreamId, TensorId, TensorData),
    RegisterTensorRemote(TensorRemote, TensorId),
    ExposeTensorRemote {
        tensor: TensorIr,
        count: u32,
        transfer_id: TensorTransferId,
    },
    /// Source side of a same-host transfer: hand the device-resident primitive for `tensor`
    /// to the server's local transfer registry under `transfer_id`. No host readback — the
    /// counterpart [`RegisterTensorLocal`](ComputeTask::RegisterTensorLocal), running on the
    /// target session of the same server, moves it onto the target device via the inner
    /// backend's `to_device`.
    ExposeTensorLocal {
        tensor: TensorIr,
        transfer_id: TensorTransferId,
    },
    /// Target side of a same-host transfer: wait for `transfer_id` to be exposed, move the
    /// primitive onto this session's device, and register it under `new_id`.
    RegisterTensorLocal {
        transfer_id: TensorTransferId,
        new_id: TensorId,
    },
    ReadTensor(RequestId, StreamId, TensorIr),
    SyncBackend(RequestId, StreamId),
    DTypeUsage(RequestId, DType),
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub struct TaskResponse {
    pub content: TaskResponseContent,
    pub id: RequestId,
}

#[allow(missing_docs)]
#[derive(Serialize, Deserialize, Debug)]
pub enum TaskResponseContent {
    /// Server responds with the selected device's settings plus the total number of devices
    /// it hosts (so the client can enumerate them, see [`RemoteDevice::enumerate`]).
    Init(DeviceSettings, u32),
    ReadTensor(Result<TensorData, ExecutionError>),
    SyncBackend(Result<(), ExecutionError>),
    DTypeUsage(DTypeUsageSet),
}
