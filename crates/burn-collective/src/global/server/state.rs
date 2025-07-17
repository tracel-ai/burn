use std::collections::HashMap;

use crate::{
    AllReduceStrategy,
    global::{
        NodeId,
        shared::{
            CentralizedAllReduceStrategy, CollectiveMessageResponse, RemoteRequest, RemoteResponse,
            RequestId, SessionId, GlobalCollectiveError,
        },
    },
};
use burn_communication::Address;
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
    node_addresses: HashMap<NodeId, Address>,
    /// The params of the current all-reduce operation, as defined by the first caller
    cur_all_reduce_strategy: Option<AllReduceStrategy>,

    /// How many total nodes for the currrent register operation, as defined by the first caller
    cur_num_nodes: Option<u32>,
    num_global_devices: u32,

    all_reduce_requests: Vec<(SessionId, RequestId, Address)>,
    register_requests: Vec<(SessionId, RequestId)>,

    sessions: HashMap<SessionId, Session>,
}

impl GlobalCollectiveState {
    pub fn new() -> Self {
        Self {
            registered_nodes: HashMap::new(),
            node_addresses: HashMap::new(),
            cur_all_reduce_strategy: None,
            cur_num_nodes: None,
            num_global_devices: 0,
            all_reduce_requests: Vec::new(),
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

    pub(crate) async fn process(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        request: RemoteRequest,
    ) {
        if let Err(err) = match request {
            RemoteRequest::AllReduce { strategy } => {
                self.all_reduce(session_id, request_id, strategy).await
            }
            RemoteRequest::Register {
                node_id,
                node_addr,
                num_nodes,
                num_devices,
            } => {
                self.register(
                    session_id,
                    request_id,
                    node_id,
                    node_addr,
                    num_nodes,
                    num_devices,
                )
                .await
            }
            RemoteRequest::Finish => self.finish(session_id, request_id).await,
        } {
            // Error occured, send it as response
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

        let mut register_requests = vec![];
        core::mem::swap(&mut register_requests, &mut self.register_requests);
        for (session, req) in register_requests {
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
                self.register_requests.push((session, req));
            }
        }

        let mut all_reduce_requests = vec![];
        core::mem::swap(&mut all_reduce_requests, &mut self.all_reduce_requests);
        for (session, req, addr) in all_reduce_requests {
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
                self.all_reduce_requests.push((session, req, addr));
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
        node_id: NodeId,
        node_addr: Address,
        num_nodes: u32,
        num_devices: u32,
    ) -> Result<(), GlobalCollectiveError> {
        if self.node_addresses.contains_key(&node_id)
            || self.registered_nodes.contains_key(&session_id)
        {
            return Err(GlobalCollectiveError::MultipleRegister(node_id));
        }
        self.registered_nodes.insert(session_id, node_id);
        self.node_addresses.insert(node_id, node_addr);

        self.register_requests.push((session_id, request_id));

        self.num_global_devices += num_devices;
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

        if self.registered_nodes.len() == num_nodes as usize {
            let mut callbacks = vec![];
            core::mem::swap(&mut callbacks, &mut self.register_requests);

            for (session, request) in callbacks {
                let content = RemoteResponse::RegisterAck {
                    num_global_devices: self.num_global_devices,
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

    async fn all_reduce(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        strategy: AllReduceStrategy,
    ) -> Result<(), GlobalCollectiveError> {
        let node_id = *self
            .registered_nodes
            .get(&session_id)
            .ok_or(GlobalCollectiveError::AllReduceBeforeRegister)?;

        if self.all_reduce_requests.is_empty() || self.cur_all_reduce_strategy.is_none() {
            self.cur_all_reduce_strategy = Some(strategy);
        } else if *self.cur_all_reduce_strategy.as_ref().unwrap() != strategy {
            log::error!(
                "Trying to all_reduce a different way ({:?}) than is currently
                    being done ({:?})",
                strategy,
                self.cur_all_reduce_strategy,
            );
            return Err(GlobalCollectiveError::AllReduceParamsMismatch);
        }

        let node_address = self.node_addresses.get(&node_id).unwrap().clone();
        self.all_reduce_requests
            .push((session_id, request_id, node_address));

        let tensor_count = self.all_reduce_requests.len();
        if tensor_count > 0 && tensor_count == self.registered_nodes.len() {
            if tensor_count == 1 {
                log::warn!(
                    "all-reduce should never be called with only one tensor, this is a no-op"
                );
            }

            // all registered callers have sent a tensor to all_reduce
            let mut requests = vec![];
            core::mem::swap(&mut requests, &mut self.all_reduce_requests);

            let mut requests_iter = requests.iter();
            let central_node = requests_iter.next().unwrap().2.clone();
            let other_nodes: Vec<Address> =
                requests_iter.map(|(_, _, addr)| addr.clone()).collect();

            for (i, (session, request, _)) in requests.iter().enumerate() {
                let is_first = i == 0;
                let strategy = if is_first {
                    CentralizedAllReduceStrategy::Central {
                        other_nodes: other_nodes.clone(),
                    }
                } else {
                    CentralizedAllReduceStrategy::Peripheral {
                        central_node: central_node.clone(),
                    }
                };
                let resp = CollectiveMessageResponse {
                    request_id: *request,
                    content: RemoteResponse::CentralizedAllReduceStrategy(strategy),
                };
                self.respond(*session, resp).await;
            }
        }

        Ok(())
    }
}
