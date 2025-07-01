use std::sync::atomic::AtomicU32;

use burn_common::id::IdGenerator;
use serde::{Deserialize, Serialize};

use crate::GlobalAllReduceParams;

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

/// Allows nodes to find each other
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeAddress(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Message {
    Init(SessionId),
    Request(RequestId, RemoteRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct MessageResponse {
    pub id: RequestId,
    pub content: RemoteResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum RemoteRequest {
    AllReduce {
        params: GlobalAllReduceParams,
    },
    Register {
        node_id: u32,
        node_addr: NodeAddress,
        num_nodes: u32,
    },
    Reset,
    /// Unregister node
    Finish,
}

// Centralized all-reduce: one node aggregates all and redistributes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum CentralizedAllReduceStrategy {
    /// For the central node
    Central { other_nodes: Vec<NodeAddress> },
    /// For non-central nodes
    Peripheral { central_node: NodeAddress },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RemoteResponse {
    RegisterAck,
    FinishAck,
    AllReduceStrategy(CentralizedAllReduceStrategy),
    Error(String),
}
