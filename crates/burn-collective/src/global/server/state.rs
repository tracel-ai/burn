use std::collections::HashMap;

use crate::{
    global::shared::{MessageResponse, NodeId, RemoteRequest, RemoteResponse, RequestId, SessionId}, AggregateParams
};
use burn_ir::TensorId;
use tokio::sync::mpsc::{Sender, Receiver};

pub(crate) struct Session {
    response_sender: Sender<MessageResponse>,
    response_receiver: Option<Receiver<MessageResponse>>,
}

impl Session {
    fn new() -> Self {
        let (response_sender, recv) = tokio::sync::mpsc::channel::<MessageResponse>(1);
        Self {
            response_sender, response_receiver: Some(recv)
        }
    }

    async fn respond(&mut self, response: MessageResponse) {
        self.response_sender.send(response).await.unwrap();
    }
}

pub(crate) struct GlobalCollectiveState {
    /// The ids passed to each register so far
    registered_nodes: Vec<NodeId>,
    /// Ids of the tensors passed so far
    tensors: Vec<TensorId>,
    /// The params of the current operation, as defined by the first caller
    cur_params: Option<AggregateParams>,
    
    _aggregate_callbacks: Vec<(SessionId, MessageResponse)>,
    register_callbacks: Vec<(SessionId, MessageResponse)>,

    sessions: HashMap<SessionId, Session>,
}

impl GlobalCollectiveState {
    pub fn new() -> Self {
        Self {
            registered_nodes: vec![],
            tensors: vec![],
            cur_params: None,
            _aggregate_callbacks: Vec::new(),
            register_callbacks: Vec::new(),
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

    pub(crate) async fn process(&mut self, session_id: SessionId, request_id: RequestId, request: RemoteRequest) {
        match request {
            RemoteRequest::Aggregate { tensor, params } => {
                self.aggregate(session_id, request_id, tensor, params).await;
            }
            RemoteRequest::Register { node_id, num_nodes } => {
                self.register(session_id, request_id, node_id, num_nodes).await;
            }
            RemoteRequest::Reset => todo!(),
        }
    }

    async fn register(&mut self, session_id: SessionId, request_id: RequestId, node_id: NodeId, num_nodes: u32) {
        if self.registered_nodes.contains(&node_id) {
            panic!("Cannot register a node twice!");
        }
        self.registered_nodes.push(node_id);

        let resp = MessageResponse {
            id: request_id, content: RemoteResponse::RegisterAck,
        };
        self.register_callbacks.push((session_id, resp));

        if self.registered_nodes.len() == num_nodes as usize {
            let mut callbacks = vec![];
            core::mem::swap(&mut callbacks, &mut self.register_callbacks);

            for (session, resp) in callbacks {
                self.respond(session, resp).await;
            }
        }
    }

    async fn aggregate(&mut self, _session_id: SessionId, _request_id: RequestId, tensor: TensorId, params: AggregateParams) {
        if self.tensors.is_empty() || self.cur_params.is_none() {
            self.cur_params = Some(params);
        } else if *self.cur_params.as_ref().unwrap() != params {
            panic!(
                "Trying to aggregate a different way ({:?}) than is currently
                    being done ({:?})",
                params, self.cur_params,
            );
        }

        self.tensors.push(tensor);

        // TODO
        // let resp = MessageResponse {
        //     id: request_id, content: todo!(),
        // };
        // self.aggregate_callbacks.push((session_id, resp));

        let tensor_count = self.tensors.len();
        if tensor_count > 0 && tensor_count == self.registered_nodes.len() {
            // all registered callers have sent a tensor to aggregate
            self.do_aggregation();
        }
    }

    // TODO rename
    fn do_aggregation(&mut self) {
        // TODO drain aggregate_callbacks
        todo!()
    }
}
