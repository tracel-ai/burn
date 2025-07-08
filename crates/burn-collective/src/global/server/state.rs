use std::collections::HashMap;

use crate::{
    GlobalAllReduceParams,
    global::shared::base::{
        CentralizedAllReduceStrategy, MessageResponse, NodeAddress, RemoteRequest, RemoteResponse,
        RequestId, SessionId,
    },
};
use tokio::sync::mpsc::{Receiver, Sender};

pub(crate) struct Session {
    response_sender: Sender<MessageResponse>,
    response_receiver: Option<Receiver<MessageResponse>>,
}

impl Session {
    fn new() -> Self {
        let (response_sender, recv) = tokio::sync::mpsc::channel::<MessageResponse>(1);
        Self {
            response_sender,
            response_receiver: Some(recv),
        }
    }

    async fn respond(&mut self, response: MessageResponse) {
        self.response_sender.send(response).await.unwrap();
    }
}

pub(crate) struct GlobalCollectiveState {
    /// The ids passed to each register so far, and their addresses
    registered_nodes: HashMap<SessionId, u32>,
    node_addresses: HashMap<u32, NodeAddress>,
    /// The params of the current operation, as defined by the first caller
    cur_params: Option<GlobalAllReduceParams>,
    num_global_devices: u32,

    all_reduce_requests: Vec<(SessionId, RequestId, NodeAddress)>,
    register_requests: Vec<(SessionId, RequestId)>,

    sessions: HashMap<SessionId, Session>,
}

impl GlobalCollectiveState {
    pub fn new() -> Self {
        Self {
            registered_nodes: HashMap::new(),
            node_addresses: HashMap::new(),
            cur_params: None,
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
    pub(crate) fn get_session_responder(&mut self, id: SessionId) -> Receiver<MessageResponse> {
        self.init_session(id);
        let session = self.sessions.get_mut(&id).unwrap();
        let response_recv = session.response_receiver.take();

        response_recv.unwrap()
    }

    pub(crate) async fn respond(&mut self, session_id: SessionId, response: MessageResponse) {
        let session = self.sessions.get_mut(&session_id).unwrap();
        session.respond(response).await;
    }

    pub(crate) async fn process(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        request: RemoteRequest,
    ) {
        match request {
            RemoteRequest::AllReduce { params } => {
                self.all_reduce(session_id, request_id, params).await;
            }
            RemoteRequest::Register {
                node_id,
                node_addr,
                num_nodes,
                num_local_devices,
            } => {
                self.register(
                    session_id,
                    request_id,
                    node_id,
                    node_addr,
                    num_nodes,
                    num_local_devices,
                )
                .await;
            }
            RemoteRequest::Reset => self.reset(),
            RemoteRequest::Finish => self.finish(session_id, request_id).await,
        }
    }

    fn reset(&mut self) {
        self.registered_nodes.clear();
        self.node_addresses.clear();
        self.cur_params = None;
        self.num_global_devices = 0;
        self.all_reduce_requests.clear();
        self.register_requests.clear();
        self.sessions.clear();
    }

    /// Un-register a node. Any pending requests will be cancelled, returning error responses.
    async fn finish(&mut self, session_id: SessionId, request_id: RequestId) {
        let node_id = self
            .registered_nodes
            .remove(&session_id)
            .unwrap_or_else(|| {
                panic!("Cannot finish the session {session_id:?} that was not registered");
            });
        self.node_addresses.remove(&node_id);

        let mut register_requests = vec![];
        core::mem::swap(&mut register_requests, &mut self.register_requests);
        for (session, req) in register_requests {
            if session == session_id {
                // Send a response if we are finishing a session with a pending register request
                let content = RemoteResponse::Error("Register cancelled by a Finish".to_string());
                let response = MessageResponse { id: req, content };
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
                let content = RemoteResponse::Error("All-Reduce cancelled by a Finish".to_string());
                let response = MessageResponse { id: req, content };
                self.respond(session_id, response).await;
            } else {
                // keep the register request
                self.all_reduce_requests.push((session, req, addr));
            }
        }

        self.respond(
            session_id,
            MessageResponse {
                id: request_id,
                content: RemoteResponse::FinishAck,
            },
        )
        .await;
    }

    async fn register(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        node_id: u32,
        node_addr: NodeAddress,
        num_nodes: u32,
        num_devices: u32,
    ) {
        if self.node_addresses.contains_key(&node_id)
            || self.registered_nodes.contains_key(&session_id)
        {
            panic!("Cannot register a node twice! Node id: {node_id}");
        }
        self.registered_nodes.insert(session_id, node_id);
        self.node_addresses.insert(node_id, node_addr);

        self.register_requests.push((session_id, request_id));

        self.num_global_devices += num_devices;

        if self.registered_nodes.len() == num_nodes as usize {
            let mut callbacks = vec![];
            core::mem::swap(&mut callbacks, &mut self.register_requests);

            for (session, request) in callbacks {
                let content = RemoteResponse::RegisterAck {
                    num_global_devices: self.num_global_devices,
                };
                let resp = MessageResponse {
                    id: request,
                    content,
                };
                self.respond(session, resp).await;
            }
        }
    }

    async fn all_reduce(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        params: GlobalAllReduceParams,
    ) {
        let node_id = match self.registered_nodes.get(&session_id) {
            Some(node_id) => *node_id,
            None => panic!("Cannot all_reduce without having registered!"),
        };

        if self.all_reduce_requests.is_empty() || self.cur_params.is_none() {
            self.cur_params = Some(params);
        } else if *self.cur_params.as_ref().unwrap() != params {
            panic!(
                "Trying to all_reduce a different way ({:?}) than is currently
                    being done ({:?})",
                params, self.cur_params,
            );
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
            let other_nodes: Vec<NodeAddress> =
                requests_iter.map(|(_, _, addr)| addr.clone()).collect();

            for (i, (session, request, addr)) in requests.iter().enumerate() {
                let is_first = i == 0;
                let strategy = if is_first {
                    eprintln!("Central is: {addr:?}");
                    CentralizedAllReduceStrategy::Central {
                        other_nodes: other_nodes.clone(),
                    }
                } else {
                    CentralizedAllReduceStrategy::Peripheral {
                        central_node: central_node.clone(),
                    }
                };
                let resp = MessageResponse {
                    id: *request,
                    content: RemoteResponse::CentralizedAllReduceStrategy(strategy),
                };
                self.respond(*session, resp).await;
            }
        }
    }
}
