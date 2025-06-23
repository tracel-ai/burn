use std::collections::HashMap;

use crate::{
    GlobalAggregateParams,
    global::shared::{
        CentralizedAggregateStrategy, MessageResponse, NodeAddress, NodeId, RemoteRequest,
        RemoteResponse, RequestId, SessionId,
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
    registered_nodes: HashMap<SessionId, NodeId>,
    node_addresses: HashMap<NodeId, NodeAddress>,
    /// The params of the current operation, as defined by the first caller
    cur_params: Option<GlobalAggregateParams>,

    aggregate_requests: Vec<(SessionId, RequestId, NodeAddress)>,
    register_requests: Vec<(SessionId, RequestId)>,

    sessions: HashMap<SessionId, Session>,
}

impl GlobalCollectiveState {
    pub fn new() -> Self {
        Self {
            registered_nodes: HashMap::new(),
            node_addresses: HashMap::new(),
            cur_params: None,
            aggregate_requests: Vec::new(),
            register_requests: Vec::new(),
            sessions: HashMap::new(),
        }
    }

    /// Create the session with given id if necessary, and get the response receiver
    pub(crate) fn get_session_responder(&mut self, id: SessionId) -> Receiver<MessageResponse> {
        let mut session = match self.sessions.remove(&id) {
            Some(val) => val,
            None => Session::new(),
        };
        let response_recv = session.response_receiver.take();
        self.sessions.insert(id, session);

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
            RemoteRequest::Aggregate { params } => {
                self.aggregate(session_id, request_id, params).await;
            }
            RemoteRequest::Register {
                node_id,
                node_addr,
                num_nodes,
            } => {
                self.register(session_id, request_id, node_id, node_addr, num_nodes)
                    .await;
            }
            RemoteRequest::Reset => todo!(),
        }
    }

    async fn register(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        node_id: NodeId,
        node_addr: NodeAddress,
        num_nodes: u32,
    ) {
        if self.node_addresses.contains_key(&node_id)
            || self.registered_nodes.contains_key(&session_id)
        {
            panic!("Cannot register a node twice!");
        }
        self.registered_nodes.insert(session_id, node_id);
        self.node_addresses.insert(node_id, node_addr);

        self.register_requests.push((session_id, request_id));

        if self.registered_nodes.len() == num_nodes as usize {
            let mut callbacks = vec![];
            core::mem::swap(&mut callbacks, &mut self.register_requests);

            for (session, request) in callbacks {
                let resp = MessageResponse {
                    id: request,
                    content: RemoteResponse::RegisterAck,
                };
                self.respond(session, resp).await;
            }
        }
    }

    async fn aggregate(
        &mut self,
        session_id: SessionId,
        request_id: RequestId,
        params: GlobalAggregateParams,
    ) {
        let node_id = match self.registered_nodes.get(&session_id) {
            Some(node_id) => *node_id,
            None => panic!("Cannot aggregate without having registered!"),
        };

        if self.aggregate_requests.is_empty() || self.cur_params.is_none() {
            self.cur_params = Some(params);
        } else if *self.cur_params.as_ref().unwrap() != params {
            panic!(
                "Trying to aggregate a different way ({:?}) than is currently
                    being done ({:?})",
                params, self.cur_params,
            );
        }

        let node_address = self.node_addresses.get(&node_id).unwrap().clone();
        self.aggregate_requests
            .push((session_id, request_id, node_address));

        let tensor_count = self.aggregate_requests.len();
        if tensor_count > 0 && tensor_count == self.registered_nodes.len() {
            // all registered callers have sent a tensor to aggregate
            let mut requests = vec![];
            core::mem::swap(&mut requests, &mut self.aggregate_requests);

            let mut requests_iter = requests.iter();
            let central_node = requests_iter.next().unwrap().2.clone();
            let other_nodes: Vec<NodeAddress> =
                requests_iter.map(|(_, _, addr)| addr.clone()).collect();

            for (i, (session, request, _)) in requests.iter().enumerate() {
                let is_first = i == 0;
                let strategy = if is_first {
                    CentralizedAggregateStrategy::Central {
                        other_nodes: other_nodes.clone(),
                    }
                } else {
                    CentralizedAggregateStrategy::Peripheral {
                        central_node: central_node.clone(),
                    }
                };
                let resp = MessageResponse {
                    id: *request,
                    content: RemoteResponse::AggregateStrategy(strategy),
                };
                self.respond(*session, resp).await;
            }
        }
    }
}
