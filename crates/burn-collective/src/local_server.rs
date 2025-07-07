use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, SyncSender},
    },
};

use burn_network::websocket::{WsClient, WsServer};
use burn_tensor::backend::Backend;
use tokio::runtime::{Builder, Runtime};

use crate::{
    AllReduceParams, AllReduceStrategy, GlobalAllReduceParams, RegisterParams,
    centralized::all_reduce_centralized,
    global::client::{base::GlobalCollectiveClient, data_server::TensorDataService},
    ring::all_reduce_ring,
    tree::all_reduce_tree,
};

// Define the client/server communication on the network
type Client = WsClient;
type Server<B> = WsServer<Arc<TensorDataService<B, Client>>>;

/// The local collective server that manages all the collective aggregation operations
/// (like all-reduce) between local threads.
/// This thread takes in messages from different clients. The clients must register, than they can
/// send an aggregate message. They must all use the same parameters for the same aggregate
/// operation.
pub struct LocalCollectiveServer<B: Backend> {
    /// Channel receiver for messages from clients
    message_rec: Receiver<Message<B>>,

    /// The ids passed to each register so far
    registered_ids: Vec<u32>,
    /// The params of the current operation, as defined by the first caller
    cur_register_params: Option<RegisterParams>,
    /// The params of the current operation, as defined by the first caller
    cur_allreduce_params: Option<AllReduceParams>,
    /// The tensor primitives passed by each operation call
    tensors: Vec<B::FloatTensorPrimitive>,
    /// Callbacks for when all registers are done
    callbacks_register: Vec<SyncSender<()>>,
    /// Callbacks for when aggregate is done
    callbacks_allreduce: Vec<SyncSender<B::FloatTensorPrimitive>>,

    /// Client for global collective operations
    global_client: Option<GlobalCollectiveClient<B, Client, Server<B>>>,
}

#[derive(Debug)]
pub(crate) enum Message<B: Backend> {
    AllReduce {
        tensor: B::FloatTensorPrimitive,
        params: AllReduceParams,
        callback: SyncSender<B::FloatTensorPrimitive>,
    },
    Register {
        id: u32,
        params: RegisterParams,
        callback: SyncSender<()>,
    },
    Reset,
    Finish {
        id: u32,
        callback: SyncSender<()>,
    },
}

#[derive(Clone)]
pub(crate) struct LocalCollectiveClient<B: Backend> {
    channel: SyncSender<Message<B>>,
    _runtime: Arc<Runtime>,
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

    pub fn register(&mut self, id: u32, params: RegisterParams) {
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

    pub fn all_reduce(
        &self,
        tensor: B::FloatTensorPrimitive,
        params: AllReduceParams,
    ) -> B::FloatTensorPrimitive {
        let (callback, rec) = std::sync::mpsc::sync_channel::<B::FloatTensorPrimitive>(1);

        self.channel
            .send(Message::AllReduce {
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

    pub fn finish(&self, id: u32) {
        let (callback, rec) = std::sync::mpsc::sync_channel::<()>(1);
        self.channel.send(Message::Finish { id, callback }).unwrap();

        rec.recv()
            .expect("Failed to receive response from collective server for finish operation")
    }
}

impl<B: Backend> LocalCollectiveServer<B> {
    fn new(rec: Receiver<Message<B>>) -> Self {
        Self {
            message_rec: rec,
            registered_ids: vec![],
            cur_register_params: None,
            cur_allreduce_params: None,
            tensors: vec![],
            callbacks_register: vec![],
            callbacks_allreduce: vec![],
            global_client: None,
        }
    }

    /// Starts the local collective server thread
    pub(crate) fn start() -> LocalCollectiveClient<B> {
        let (sender, rec) = std::sync::mpsc::sync_channel::<Message<B>>(50);

        let _runtime = Arc::new(Builder::new_multi_thread().enable_all().build().unwrap());

        _runtime.spawn(async {
            let mut aggregator = LocalCollectiveServer::new(rec);

            while let Ok(message) = aggregator.message_rec.recv() {
                aggregator.process_message(message).await;
            }

            panic!("Aggregator message failed");
        });

        LocalCollectiveClient {
            channel: sender,
            _runtime,
        }
    }

    async fn process_message(&mut self, message: Message<B>) {
        match message {
            Message::AllReduce {
                tensor,
                params,
                callback,
            } => {
                self.process_all_reduce_message(tensor, params, callback)
                    .await
            }
            Message::Register {
                id,
                params,
                callback,
            } => self.process_register_message(id, params, callback).await,
            Message::Reset => self.reset(),
            Message::Finish { id, callback } => {
                self.process_finish_message(id, callback).await;
            }
        }
    }

    fn reset(&mut self) {
        self.registered_ids.clear();
        self.tensors.clear();
        self.cur_allreduce_params = None;
    }

    async fn process_finish_message(&mut self, id: u32, callback: SyncSender<()>) {
        if !self.registered_ids.contains(&id) {
            panic!("Cannot un-register a node twice!");
        }

        // Remove registered with id
        self.registered_ids.retain(|&x| x != id);

        if self.registered_ids.is_empty() {
            if let Some(global_client) = self.global_client.as_mut() {
                global_client.finish().await;
            }
        }

        callback.send(()).unwrap();
    }

    async fn process_all_reduce_message(
        &mut self,
        tensor: <B as Backend>::FloatTensorPrimitive,
        params: AllReduceParams,
        callback: SyncSender<<B as Backend>::FloatTensorPrimitive>,
    ) {
        if self.tensors.is_empty() || self.cur_allreduce_params.is_none() {
            self.cur_allreduce_params = Some(params);
        } else if self.cur_allreduce_params.clone().unwrap() != params {
            panic!(
                "Trying to aggregate a different way ({:?}) than is currently
                    being done ({:?})",
                params, self.cur_allreduce_params,
            );
        }

        self.tensors.push(tensor);
        self.callbacks_allreduce.push(callback);

        let tensor_count = self.tensors.len();
        if tensor_count > 0 && tensor_count == self.registered_ids.len() {
            // all registered callers have sent a tensor to aggregate
            self.all_reduce().await
        }
    }

    async fn process_register_message(
        &mut self,
        id: u32,
        params: RegisterParams,
        callback: SyncSender<()>,
    ) {
        if self.registered_ids.contains(&id) {
            panic!("Cannot register a node twice!");
        }
        if self.registered_ids.is_empty() || self.cur_register_params.is_none() {
            self.cur_register_params = Some(params.clone());
        } else if self.cur_register_params.clone().unwrap() != params {
            panic!(
                "Trying to register a different way ({:?}) than is currently
                        being done ({:?})",
                params, self.cur_allreduce_params,
            );
        }

        self.registered_ids.push(id);
        self.callbacks_register.push(callback);

        if let Some(global_params) = &params.global_params {
            if self.global_client.is_none() {
                let client = GlobalCollectiveClient::new(
                    &global_params.server_url,
                    &global_params.client_url,
                    global_params.client_data_port,
                );
                self.global_client = Some(client)
            }
        }

        // All have registered, callback
        if self.registered_ids.len() == params.num_devices as usize {
            if let Some(global_params) = &params.global_params {
                let client = self.global_client.as_mut().unwrap();
                client
                    .register(global_params.node_id, global_params.clone())
                    .await;
            }

            for callback in self.callbacks_register.drain(..) {
                callback.send(()).unwrap();
            }
        }
    }

    async fn all_reduce(&mut self) {
        let params = self.cur_allreduce_params.as_ref().unwrap();
        let kind = &params.kind;
        let local_strategy = &params.local_strategy;
        let global_strategy = &params.global_strategy;
        let tensor_count = self.tensors.len();

        let mut outs = match local_strategy {
            AllReduceStrategy::Centralized => {
                let out = all_reduce_centralized::<B>(&mut self.tensors, kind);
                vec![out; tensor_count]
            }
            AllReduceStrategy::Tree(arity) => all_reduce_tree::<B>(&mut self.tensors, kind, *arity),
            AllReduceStrategy::Ring => all_reduce_ring::<B>(&mut self.tensors, kind),
        };

        if let Some(global_client) = &mut self.global_client {
            let mut tensor = outs.remove(0);
            let params = GlobalAllReduceParams {
                strategy: global_strategy.unwrap(),
            };

            let device = B::float_device(&tensor);
            tensor = global_client.all_reduce(tensor, params, &device).await;
            // replace all results with global aggregation result
            let len = outs.len() + 1;
            outs = vec![tensor; len];
        }

        // Callbacks return results
        for callback in self.callbacks_allreduce.drain(..) {
            let out = outs.remove(0);
            callback.send(out).unwrap();
        }
    }
}
