use std::{collections::HashMap, sync::atomic::AtomicU32};

use crate::{NodeId, PeerId};
use burn_communication::{Address, CommunicationError};
use burn_std::id::IdGenerator;
use serde::{Deserialize, Serialize};

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

/// Unique identifier that can represent a session between a node and the orchestrator.
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
        /// Endpoint for this node
        node_addr: Address,
        /// Number of total nodes
        num_nodes: u32,
        /// List of peers on this node
        peers: Vec<PeerId>,
    },

    /// Unregister node
    Finish,
}

/// Responses for each server request
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RemoteResponse {
    /// Response to a register request
    Register {
        /// The orchestrator gives the node its id
        node_id: NodeId,
        /// All the nodes in the collective: including self
        nodes: HashMap<NodeId, Address>,
        /// How many devices exist globally? For averaging values
        num_global_devices: u32,
    },

    // Finish
    FinishAck,

    // There was a server-side error
    Error(GlobalCollectiveError),
}

/// Errors that occur during collective operations on the global level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlobalCollectiveError {
    /// Operations that can't be done before registering
    AllReduceBeforeRegister,
    /// Ring all-reduce can't be done if all tensor dimensions are smaller than the number of nodes.
    RingReduceImpossible,

    /// Either a node has unregistered twice, or a Finish has been called before a Register
    NotRegisteredOnFinish,
    /// Finish has been called before a Register operation was finished
    PendingRegisterOnFinish,
    /// Trying to register a different way than is currently being done
    RegisterParamsMismatch,
    /// Trying to register while already registered
    DoubleRegister,
    /// Trying to aggregate a different way than is currently being done
    AllReduceParamsMismatch,

    /// First message on socket should be Message::Init
    FirstMsgNotInit,
    /// Messages should be rmp_serde serialized `Message` types
    InvalidMessage,
    /// A peer behaved unexpectedly
    PeerSentIncoherentTensor,
    /// Tried to download from a peer, but the peer closed or lost the connection
    PeerLost(NodeId),
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
