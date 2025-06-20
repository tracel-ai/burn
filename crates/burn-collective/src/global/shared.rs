use std::sync::atomic::AtomicU32;

use burn_common::id::IdGenerator;
use burn_ir::TensorId;
use serde::{Deserialize, Serialize};

use crate::AggregateParams;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(crate) struct RequestId(u32);

static ID_COUNTER: AtomicU32 = AtomicU32::new(0);
impl RequestId {
    pub(crate) fn new() -> Self {
        let id = ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(id)
    }
}

/// Unique identifier that can represent a session.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub(crate) struct SessionId {
    id: u64,
}

impl SessionId {
    /// Create a new [session id](SessionId).
    pub(crate) fn new() -> Self {
        Self {
            id: IdGenerator::generate(),
        }
    }
}

/// Unique identifier for each node
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub(crate) struct NodeId {
    id: u64,
}

impl NodeId {
    pub(crate) fn new() -> Self {
        Self {
            id: IdGenerator::generate(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Message {
    Init(SessionId),
    Request(RequestId, RemoteRequest)
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MessageResponse {
    pub id: RequestId,
    pub content: RemoteResponse
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum RemoteRequest {
    Aggregate {
        tensor: TensorId,
        params: AggregateParams,
    },
    Register {
        node_id: NodeId,
        num_nodes: u32,
    },
    Reset,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RemoteResponse {
    RegisterAck,
    // Error(String),
}
