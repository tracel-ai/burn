use burn_backend::{DTypeUsageSet, ExecutionError, TensorData};
use burn_communication::{Address, external_comm::TensorTransferId};
use burn_ir::{GraphBindings, GraphId, OperationIr, TensorId, TensorIr};
use burn_std::{
    DType, DeviceSettings,
    id::{IdGenerator, StreamId},
};
use serde::{Deserialize, Serialize};
use std::fmt::Display;

/// Routing id for a task whose result is fetched back.
///
/// Only the result-producing tasks ([`Task::ReadTensor`], [`Task::SyncBackend`],
/// [`Task::DTypeUsage`]) carry a `RequestId`; the server echoes it on its [`TaskResponse`] so
/// the client demultiplexes results back to the right pending callback. Fire-and-forget tasks
/// have no id because no result ever comes back. Collective ops (all-reduce, sync-collective)
/// are plain fire-and-forget [`OperationIr`]s carried by [`Task::RegisterOperation`].
pub type RequestId = u64;

/// Unique identifier that can represent a session.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct SessionId {
    id: u64,
}

impl Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SessionId({})", self.id)
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

/// A single message on a session's `/submit` (or handshake) stream: either a session-lifecycle
/// signal (`Init`/`Close`) or a [`Task`] to run.
#[allow(missing_docs, clippy::large_enum_variant)]
#[derive(Serialize, Deserialize, Debug)]
pub enum RemoteMessage {
    /// A unit of work to run within the bound session.
    Task(Task),
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
pub enum Task {
    Seed(u64),
    /// A single [`OperationIr`] tagged with the stream it was issued on.
    ///
    /// Each op is sent independently — batching across streams would lose stream
    /// identity, and the server-side backend (fusion, etc.) handles any batching of
    /// its own.
    RegisterOperation(StreamId, OperationIr),
    /// Register a reusable group of operations under `graph_id` (relative form) *and* immediately
    /// execute it with `bindings`.
    ///
    /// Registering a new graph always coincides with its first execution, so the two are combined
    /// into a single task to avoid a redundant round-trip on a cache miss. The server caches the
    /// graph (per session) and replays it on every subsequent [`ExecuteGraph`](Task::ExecuteGraph).
    /// This is how the fusion-enabled client avoids re-sending a recurring op sequence (e.g. a
    /// model block) on every step.
    RegisterAndExecuteGraph {
        stream_id: StreamId,
        graph_id: GraphId,
        relative_graph: Vec<OperationIr>,
        bindings: GraphBindings,
    },
    /// Replay an already-registered graph with the given concrete bindings.
    ExecuteGraph {
        stream_id: StreamId,
        graph_id: GraphId,
        bindings: GraphBindings,
    },
    RegisterTensor(StreamId, TensorId, TensorData),
    /// Register `new_id` as an alias of `src_id` — a second server handle over the same buffer.
    ///
    /// Emitted by the fusion layer's cross-stream sharing so a tensor used on multiple streams
    /// gets one server id per stream view, each freeable independently. Ordered after the task
    /// that materializes `src_id`, like every other op on the session's FIFO worker.
    RegisterAlias {
        stream_id: StreamId,
        new_id: TensorId,
        src_id: TensorId,
    },
    RegisterTensorRemote(StreamId, TensorRemote, TensorId),
    ExposeTensorRemote {
        stream_id: StreamId,
        tensor: TensorIr,
        count: u32,
        transfer_id: TensorTransferId,
    },
    /// Source side of a same-host transfer: hand the device-resident primitive for `tensor`
    /// to the server's local comm registry under `transfer_id`. No host readback — the
    /// counterpart [`RegisterTensorLocal`](Task::RegisterTensorLocal), running on the
    /// target session of the same server, moves it onto the target device via the inner
    /// backend's `to_device`.
    ///
    /// `stream_id` is the client stream that produced `tensor`, so the server reads it back on
    /// the same stream the producing ops ran on rather than an arbitrary server thread.
    ExposeTensorLocal {
        stream_id: StreamId,
        tensor: TensorIr,
        transfer_id: TensorTransferId,
    },
    /// Target side of a same-host transfer: wait for `transfer_id` to be exposed, move the
    /// primitive onto this session's device, and register it under `new_id`.
    ///
    /// `stream_id` is the client stream that will consume `new_id`, so the registration lands
    /// on the same stream as the ops that use the transferred tensor.
    RegisterTensorLocal {
        stream_id: StreamId,
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
