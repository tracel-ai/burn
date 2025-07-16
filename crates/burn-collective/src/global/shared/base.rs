use core::ops::Range;
use std::sync::atomic::AtomicU32;

use burn_common::id::IdGenerator;
use burn_communication::Address;
use serde::{Deserialize, Serialize};

use crate::{
    AllReduceStrategy, global::server::base::GlobalCollectiveError,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RequestId(u32);

static ID_COUNTER: AtomicU32 = AtomicU32::new(0);
impl RequestId {
    pub fn new() -> Self {
        let id = ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(id)
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier that can represent a session.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct SessionId {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(u32);

impl From<u32> for NodeId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectiveMessage {
    Init(SessionId),
    Request(RequestId, RemoteRequest),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveMessageResponse {
    pub request_id: RequestId,
    pub content: RemoteResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RemoteRequest {
    AllReduce {
        strategy: AllReduceStrategy,
    },
    Register {
        node_id: NodeId,
        node_addr: Address,
        /// Number of total nodes
        num_nodes: u32,
        /// Number of local devices on the requesting node 
        num_devices: u32,
    },
    Reset,
    /// Unregister node
    Finish,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemoteResponse {
    RegisterAck { num_global_devices: u32 },
    FinishAck,
    CentralizedAllReduceStrategy(CentralizedAllReduceStrategy),
    TreeAllReduceStrategy(TreeAllReduceStrategy),
    RingAllReduceStrategy(RingAllReduceStrategy),
    Error(GlobalCollectiveError),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CentralizedAllReduceStrategy {
    /// The central node is the one that will perform the all-reduce operation
    Central { other_nodes: Vec<Address> },
    /// The peripheral nodes are the ones that will send their tensors to the central node
    Peripheral { central_node: Address },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeAllReduceStrategy {
    pub children: Vec<Address>,
    pub parent: Option<Address>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RingAllReduceStrategy {
    // Position in ring
    pub next_node: Address,

    // Slicing strategy
    pub slice_dim: usize,
    pub slice_ranges: Vec<Range<usize>>,

    // Slice ordering
    pub slice_count: usize,
    pub first_slice: usize,
}
