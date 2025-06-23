use std::sync::atomic::AtomicU32;

use burn_common::id::IdGenerator;
use serde::{Deserialize, Serialize};

use crate::GlobalAggregateParams;

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
    Aggregate {
        params: GlobalAggregateParams,
    },
    Register {
        node_id: NodeId,
        node_addr: NodeAddress,
        num_nodes: u32,
    },
    Reset,
}

// For now this is a centralized all-reduce. TODO add more strategies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum CentralizedAggregateStrategy {
    /// For the central node
    Central { other_nodes: Vec<NodeAddress> },
    /// For non-central nodes
    Peripheral { central_node: NodeAddress },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RemoteResponse {
    RegisterAck,
    AggregateStrategy(CentralizedAggregateStrategy),
    Error(String),
}
