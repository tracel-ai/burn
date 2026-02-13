use crate::global::{
    NodeId,
    shared::{
        CollectiveMessageResponse, GlobalCollectiveError, RemoteRequest, RemoteResponse, RequestId,
        SessionId,
    },
};
use burn_communication::Address;
use burn_tensor::backend::PeerId;
use std::collections::HashMap;
use tokio::sync::mpsc::{Receiver, Sender};

pub(crate) struct Session {
    response_sender: Sender<CollectiveMessageResponse>,
    response_receiver: Option<Receiver<CollectiveMessageResponse>>,
}

impl Session {
    fn new() -> Self {
        let (response_sender, recv) = tokio::sync::mpsc::channel::<CollectiveMessageResponse>(1);
        Self {
            response_sender,
            response_receiver: Some(recv),
        }
    }

    async fn respond(&mut self, response: CollectiveMessageResponse) {
        self.response_sender.send(response).await.unwrap();
    }
}

pub(crate) struct GlobalCollectiveState {
    /// The ids passed to each register so far, and their addresses
    registered_nodes: HashMap<SessionId, NodeId>,
    /// Address for each node
    node_addresses: HashMap<NodeId, Address>,
    /// Peer on each node
    node_peers: HashMap<NodeId, Vec<PeerId>>,

    /// How many total nodes for the current register operation, as defined by the first caller
    cur_num_nodes: Option<u32>,
    /// How many peers have registered total
    num_global_peers: u32,

    register_requests: Vec<(SessionId, RequestId, NodeId)>,

    sessions: HashMap<SessionId, Session>,
}

impl GlobalCollectiveState {
    pub fn new() -> Self {
        Self {
            registered_nodes: HashMap::new(),
            node_addresses: HashMap::new(),
            node_peers: HashMap::new(),
            cur_num_nodes: None,
            num_global_peers: 0,
            register_requests: Vec::new(),
            sessions: HashMap::new(),
        }
    }

    pub(crate) fn init_session(&mut self, id: SessionId) {
        if self.sessions.contains_key(&id) {
            return;
        }
        self.sessions.insert(id, Session::new());
    }

    /// Create the session with given id if necessary, and get the response receiver
    pub(crate) fn get_session_responder(
        &mut self,
        id: SessionId,
    ) -> Receiver<CollectiveMessageResponse> {
        self.init_session(id);
        let session = self.sessions.get_mut(&id).unwrap();
        let response_recv = session.response_receiver.take();

        response_recv.unwrap()
    }

    pub(crate) async fn respond(
        &mut self,
        session_id: SessionId,
        response: CollectiveMessageResponse,
    ) {
        let session = self.sessions.get_mut(&session_id).unwrap();
        session.respond(response).await;
    }

    /// Process an incoming node's request
    pub(crate) async fn process_request(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        request: RemoteRequest,
    ) {
        if let Err(err) = match request {
            RemoteRequest::Register {
                node_addr,
                num_nodes,
                peers,
            } => {
                self.register(session_id, request_id, node_addr, num_nodes, peers)
                    .await
            }
            RemoteRequest::Finish => self.finish(session_id, request_id).await,
        } {
            // Error occurred, send it as response
            let content = RemoteResponse::Error(err);
            self.respond(
                session_id,
                CollectiveMessageResponse {
                    request_id,
                    content,
                },
            )
            .await;
        }
    }

    /// Un-register a node. Any pending requests will be cancelled, returning error responses.
    async fn finish(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
    ) -> Result<(), GlobalCollectiveError> {
        let node_id = self
            .registered_nodes
            .remove(&session_id)
            .ok_or(GlobalCollectiveError::NotRegisteredOnFinish)?;
        self.node_addresses.remove(&node_id);
        self.node_peers.remove(&node_id);
        self.num_global_peers = 0;

        let mut register_requests = vec![];
        core::mem::swap(&mut register_requests, &mut self.register_requests);
        for (session, req, node_id) in register_requests {
            if session == session_id {
                // Send a response if we are finishing a session with a pending register request
                let content = RemoteResponse::Error(GlobalCollectiveError::PendingRegisterOnFinish);
                let response = CollectiveMessageResponse {
                    request_id: req,
                    content,
                };
                self.respond(session_id, response).await;
            } else {
                // keep the register request
                self.register_requests.push((session, req, node_id));
            }
        }

        self.respond(
            session_id,
            CollectiveMessageResponse {
                request_id,
                content: RemoteResponse::FinishAck,
            },
        )
        .await;

        Ok(())
    }

    async fn register(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        node_addr: Address,
        num_nodes: u32,
        peers: Vec<PeerId>,
    ) -> Result<(), GlobalCollectiveError> {
        match &self.cur_num_nodes {
            Some(cur_num_nodes) => {
                if *cur_num_nodes != num_nodes {
                    return Err(GlobalCollectiveError::RegisterParamsMismatch);
                }
            }
            None => {
                self.cur_num_nodes = Some(num_nodes);
            }
        }

        self.num_global_peers += peers.len() as u32;

        let node_id: NodeId = self.registered_nodes.len().into();
        self.registered_nodes.insert(session_id, node_id);
        if self.node_addresses.values().any(|addr| node_addr == *addr) {
            return Err(GlobalCollectiveError::DoubleRegister);
        }
        self.node_addresses.insert(node_id, node_addr);
        self.node_peers.insert(node_id, peers);

        self.register_requests
            .push((session_id, request_id, node_id));

        if self.registered_nodes.len() == num_nodes as usize {
            let mut callbacks = vec![];
            core::mem::swap(&mut callbacks, &mut self.register_requests);

            for (session, request, node_id) in callbacks {
                let content = RemoteResponse::Register {
                    node_id,
                    nodes: self.node_addresses.clone(),
                    num_global_devices: self.num_global_peers,
                };
                let resp = CollectiveMessageResponse {
                    request_id: request,
                    content,
                };
                self.respond(session, resp).await;
            }
        }

        Ok(())
    }
}
