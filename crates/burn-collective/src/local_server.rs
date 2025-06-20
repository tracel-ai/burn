use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{
        Mutex,
        mpsc::{Receiver, SyncSender},
    },
};

use burn_tensor::backend::Backend;
use tokio::runtime::Builder;

use crate::{
    centralized::all_reduce_centralized, global::{client::base::GlobalCollectiveClient, shared::NodeId}, ring::all_reduce_ring, tree::all_reduce_tree, AggregateParams, AggregateStrategy, RegisterParams
};

/// The local collective server that manages all the collective aggregation operations
/// (like all-reduce) between local threads.
/// This thread takes in messages from different clients. The clients must register, than they can
/// send an aggregate message. They must all use the same parameters for the same aggregate
/// operation.
pub struct LocalCollectiveServer<B: Backend> {
    /// Channel receiver for messages from clients
    message_rec: Receiver<Message<B>>,

    /// The ids passed to each register so far
    registered_nodes: Vec<u32>,
    /// The params of the current operation, as defined by the first caller
    cur_register_params: Option<RegisterParams>,
    /// The params of the current operation, as defined by the first caller
    cur_aggregate_params: Option<AggregateParams>,
    /// The tensor primitives passed by each operation call
    tensors: Vec<B::FloatTensorPrimitive>,
    /// Callbacks for when all registers are done
    callbacks_register: Vec<SyncSender<()>>,
    /// Callbacks for when aggregate is done
    callbacks_aggregate: Vec<SyncSender<B::FloatTensorPrimitive>>,

    /// Client for global collective operations
    global_client: Option<GlobalCollectiveClient<B>>,
}

#[derive(Debug)]
pub(crate) enum Message<B: Backend> {
    Aggregate {
        tensor: B::FloatTensorPrimitive,
        params: AggregateParams,
        callback: SyncSender<B::FloatTensorPrimitive>,
    },
    Register {
        id: u32,
        params: RegisterParams,
        callback: SyncSender<()>,
    },
    Reset,
}

#[derive(Clone)]
pub(crate) struct LocalCollectiveClient<B: Backend> {
    channel: SyncSender<Message<B>>,
}

static STATE: Mutex<Option<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = Mutex::new(None);

pub(crate) fn get_collective_client<B: Backend>() -> LocalCollectiveClient<B> {
    let mut state = STATE.lock().unwrap();

    if state.is_none() {
        *state = Some(HashMap::new());
    }
    let hashmap = state.as_mut().unwrap();

    let typeid = core::any::TypeId::of::<B>();

    let val = match hashmap.get(&typeid) {
        Some(val) => val,
        None => {
            let client = LocalCollectiveServer::start();
            hashmap.insert(typeid, Box::new(client.clone()));
            return client;
        }
    };

    val.downcast_ref().cloned().unwrap()
}

impl<B: Backend> LocalCollectiveClient<B> {
    pub fn reset(&self) {
        self.channel.send(Message::Reset).unwrap();
    }

    pub fn register(&self, id: u32, params: RegisterParams) {
        let (callback, rec) = std::sync::mpsc::sync_channel::<()>(1);

        self.channel
            .send(Message::Register {
                id,
                params,
                callback,
            })
            .unwrap();

        rec.recv().unwrap();
    }

    pub fn aggregate(
        &self,
        tensor: B::FloatTensorPrimitive,
        params: AggregateParams,
    ) -> B::FloatTensorPrimitive {
        let (callback, rec) = std::sync::mpsc::sync_channel::<B::FloatTensorPrimitive>(1);

        self.channel
            .send(Message::Aggregate {
                tensor,
                params,
                callback,
            })
            .unwrap();

        // returns a tensor primitive that may or may not be on the correct device,
        // depending on the strategy used.
        rec.recv()
            .expect("Failed to receive callback from collective server")
    }
}

impl<B: Backend> LocalCollectiveServer<B> {
    fn new(rec: Receiver<Message<B>>) -> Self {
        Self {
            message_rec: rec,
            registered_nodes: vec![],
            cur_register_params: None,
            cur_aggregate_params: None,
            tensors: vec![],
            callbacks_register: vec![],
            callbacks_aggregate: vec![],
            global_client: None,
        }
    }

    /// Starts the local collective server thread
    pub(crate) fn start() -> LocalCollectiveClient<B> {
        let (sender, rec) = std::sync::mpsc::sync_channel::<Message<B>>(50);

        let client = LocalCollectiveClient { channel: sender };

        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let _handle = std::thread::spawn(move || {
            rt.block_on(async {
                let mut aggregator = LocalCollectiveServer::new(rec);
    
                while let Ok(message) = aggregator.message_rec.recv() {
                    aggregator.process_message(message).await;
                }
    
                log::debug!("Aggregator message failed");
            })
        });

        client
    }

    async fn process_message(&mut self, message: Message<B>) {
        match message {
            Message::Aggregate {
                tensor,
                params,
                callback,
            } => self.process_aggregate_message(tensor, params, callback).await,
            Message::Register {
                id,
                params,
                callback,
            } => self.process_register_message(id, params, callback).await,
            Message::Reset => {
                self.registered_nodes.clear();
                self.tensors.clear();
                self.cur_aggregate_params = None;
            }
        }
    }

    async fn process_aggregate_message(
        &mut self,
        tensor: <B as Backend>::FloatTensorPrimitive,
        params: AggregateParams,
        callback: SyncSender<<B as Backend>::FloatTensorPrimitive>,
    ) {
        if self.tensors.is_empty() || self.cur_aggregate_params.is_none() {
            self.cur_aggregate_params = Some(params);
        } else if self.cur_aggregate_params.clone().unwrap() != params {
            panic!(
                "Trying to aggregate a different way ({:?}) than is currently
                    being done ({:?})",
                params, self.cur_aggregate_params,
            );
        }

        self.tensors.push(tensor);
        self.callbacks_aggregate.push(callback);

        let tensor_count = self.tensors.len();
        if tensor_count > 0 && tensor_count == self.registered_nodes.len() {
            // all registered callers have sent a tensor to aggregate
            self.aggregate().await
        }
    }

    async fn process_register_message(
        &mut self,
        id: u32,
        params: RegisterParams,
        callback: SyncSender<()>,
    ) {
        if self.registered_nodes.contains(&id) {
            panic!("Cannot register a node twice!");
        }
        if self.registered_nodes.is_empty() || self.cur_register_params.is_none() {
            self.cur_register_params = Some(params.clone());
        } else if self.cur_register_params.clone().unwrap() != params {
            panic!(
                "Trying to register a different way ({:?}) than is currently
                        being done ({:?})",
                params, self.cur_aggregate_params,
            );
        }

        self.registered_nodes.push(id);
        self.callbacks_register.push(callback);

        if let Some(global_params) = &params.global_params {
            let client = GlobalCollectiveClient::new(global_params.server_address.clone());
            self.global_client = Some(client)
        }

        // All have registered, callback
        if self.registered_nodes.len() == params.num_local_nodes as usize {
            if let Some(global_params) = &params.global_params {
                let client = self.global_client.as_mut().unwrap();
                let node_id = NodeId::new();
                client.register(node_id, global_params.clone()).await;
            }

            for callback in self.callbacks_register.drain(..) {
                callback.send(()).unwrap();
            }
        }
    }

    async fn aggregate(&mut self) {
        let kind = &self.cur_aggregate_params.as_ref().unwrap().kind;
        let strategy = &self.cur_aggregate_params.as_ref().unwrap().strategy;
        let tensor_count = self.tensors.len();

        let mut outs = match strategy {
            AggregateStrategy::Centralized => {
                let out = all_reduce_centralized::<B>(&mut self.tensors, kind);
                vec![out; tensor_count]
            }
            AggregateStrategy::Tree(arity) => {
                let out = all_reduce_tree::<B>(&mut self.tensors, kind, *arity);
                vec![out; tensor_count]
            }
            AggregateStrategy::Ring => all_reduce_ring::<B>(&mut self.tensors, kind),
        };

        if let Some(global_client) = &self.global_client {
            global_client.aggregate(outs.first().unwrap().clone()).await;
        }

        // Callbacks return results
        for callback in self.callbacks_aggregate.drain(..) {
            let out = outs.remove(0);
            callback.send(out).unwrap();
        }
    }
}
