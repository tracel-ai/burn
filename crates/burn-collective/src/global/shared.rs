use core::ops::Range;
use std::sync::atomic::AtomicU32;

use crate::{NodeId, PeerId};
use burn_common::id::IdGenerator;
use burn_communication::{Address, CommunicationError};
use serde::{Deserialize, Serialize};

use crate::AllReduceStrategy;

/// A unique identifier for each request made to a global orchestrator
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(crate) struct RequestId(u32);

static REQ_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
impl RequestId {
    pub(crate) fn new() -> Self {
        let id = REQ_ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(id)
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

/// Unique identifier that can represent a session between a node and a orchestrator.
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

/// Requests sent from the client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum CollectiveMessage {
    Init(SessionId),
    Request(RequestId, RemoteRequest),
}

/// Responses sent to the client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CollectiveMessageResponse {
    pub request_id: RequestId,
    pub content: RemoteResponse,
}

/// Requests made from a client to a server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum RemoteRequest {
    // Register a node
    Register {
        node_id: NodeId,
        /// Endpoint for this node
        node_addr: Address,
        /// Number of total nodes
        num_nodes: u32,
        /// List of peers on this node
        peers: Vec<PeerId>,
    },

    // Aggregate
    AllReduce {
        strategy: AllReduceStrategy,
    },

    /// Unregister node
    Finish,
}

/// Responses for each server request
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RemoteResponse {
    // Register
    RegisterAck { num_global_devices: u32 },

    // The server gives the client a strategy for all-reducing, and the client aggregates
    // independently, using the TensorDataService
    CentralizedAllReduceStrategy(CentralizedAllReduceStrategy),
    TreeAllReduceStrategy(TreeAllReduceStrategy),
    RingAllReduceStrategy(RingAllReduceStrategy),

    // Finish
    FinishAck,

    // There was a server-side error
    Error(GlobalCollectiveError),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum CentralizedAllReduceStrategy {
    /// The central node is the one that will perform the all-reduce operation
    Central { other_nodes: Vec<Address> },
    /// The peripheral nodes are the ones that will send their tensors to the central node
    Peripheral { central_node: Address },
}

/// Each node is assigned a position in a tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TreeAllReduceStrategy {
    pub children: Vec<Address>,
    pub parent: Option<Address>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RingAllReduceStrategy {
    // Position in ring
    pub next_node: Address,

    // Which dim to slice on
    pub slice_dim: usize,
    // What index ranges should correspond to each slice
    pub slice_ranges: Vec<Range<usize>>,

    // How many slices
    pub slice_count: usize,
    // What slice index should be the first to be sent
    pub first_slice: usize,
}

/// Errors that occur during collective opertaions on the global level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlobalCollectiveError {
    /// Operations that can't be done before registering
    AllReduceBeforeRegister,

    /// Can't register a node twice
    MultipleRegister(NodeId),
    /// Either a node has unregisterd twice, or a Finish has been called before a Register
    NotRegisteredOnFinish,
    /// Finish has been called before a Register operation was finished
    PendingRegisterOnFinish,
    /// Trying to register a different way than is currently being done
    RegisterParamsMismatch,
    /// Trying to aggregate a different way than is currently being done
    AllReduceParamsMismatch,

    /// First message on socket should be Message::Init
    FirstMsgNotInit,
    /// Messages should be rmp_serde serialized `Message` types
    InvalidMessage,
    /// A peer behaved unexpectedly
    PeerSentIncoherentTensor,
    /// Error from the coordinator
    Server(String),

    /// The node received an invalid response
    WrongOrchestratorResponse,
    /// Node couldn't connect to coordinator
    OrchestratorUnreachable,
}

impl<E: CommunicationError> From<E> for GlobalCollectiveError {
    fn from(err: E) -> Self {
        Self::Server(format!("{err:?}"))
    }
}
