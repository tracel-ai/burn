use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, SyncSender},
    },
};

use burn_communication::websocket::{WebSocket, WsServer};
use burn_tensor::backend::Backend;
use tokio::runtime::{Builder, Runtime};

use crate::{
    AllReduceStrategy, CollectiveError, DeviceId, GlobalRegisterParams, SharedAllReduceParams,
    centralized::all_reduce_centralized, global::client::base::GlobalCollectiveClient,
    ring::all_reduce_ring, tree::all_reduce_tree,
};

/// Define the client/server communication on the network
type Network = WebSocket;
/// Type sent to the collective client upon completion of a register request
type RegisterResult = Result<(), CollectiveError>;
/// Type sent to the collective client upon completion of a all-reduce aggregation
type AllReduceResult<T> = Result<T, CollectiveError>;
/// Type sent to the collective client upon completion of a finish request
type FinishResult = Result<(), CollectiveError>;

/// The local collective server that manages all the collective aggregation operations
/// (like all-reduce) between local threads.
/// This thread takes in messages from different clients. The clients must register, than they can
/// send an aggregate message. They must all use the same parameters for the same aggregate
/// operation.
pub(crate) struct LocalCollectiveServer<B: Backend> {
    /// Channel receiver for messages from clients
    message_rec: Receiver<Message<B>>,

    /// The ids passed to each register so far
    registered_ids: Vec<DeviceId>,
    /// The params of the current operation, as defined by the first caller
    cur_num_devices: Option<u32>,
    /// The params of the current operation, as defined by the first caller
    cur_allreduce_params: Option<SharedAllReduceParams>,
    /// The tensor primitives passed by each operation call
    tensors: Vec<B::FloatTensorPrimitive>,
    /// Callbacks for when all registers are done
    callbacks_register: Vec<SyncSender<RegisterResult>>,
    /// Callbacks for when aggregate is done
    callbacks_all_reduce: Vec<SyncSender<AllReduceResult<B::FloatTensorPrimitive>>>,

    /// Client for global collective operations
    global_client: Option<GlobalCollectiveClient<B, Network>>,
}

#[derive(Debug)]
pub(crate) enum Message<B: Backend> {
    AllReduce {
        device_id: DeviceId,
        tensor: B::FloatTensorPrimitive,
        params: SharedAllReduceParams,
        callback: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
    },
    Register {
        device_id: DeviceId,
        num_devices: u32,
        global_params: Option<GlobalRegisterParams>,
        callback: SyncSender<RegisterResult>,
    },
    Reset,
    Finish {
        id: DeviceId,
        callback: SyncSender<FinishResult>,
    },
}

#[derive(Clone)]
pub(crate) struct LocalCollectiveClient<B: Backend> {
    channel: SyncSender<Message<B>>,
}

// HashMap for each server by Backend type
static STATE: Mutex<Option<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = Mutex::new(None);

// Runtime for servers
static SERVER_RUNTIME: Mutex<Option<Arc<Runtime>>> = Mutex::new(None);

pub(crate) fn get_server_runtime() -> Arc<Runtime> {
    let mut server = SERVER_RUNTIME.lock().unwrap();
    if server.is_none() {
        // Initialize runtime
        let _runtime = Arc::new(Builder::new_multi_thread().enable_all().build().unwrap());
        *server = Some(_runtime);
    }

    server.as_ref().unwrap().clone()
}

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
            let client = LocalCollectiveServer::<B>::start();
            hashmap.insert(typeid, Box::new(client.clone()));
            return client;
        }
    };

    val.downcast_ref().cloned().unwrap()
}

impl<B: Backend> LocalCollectiveClient<B> {
    pub(crate) fn reset(&self) {
        self.channel.send(Message::Reset).unwrap();
    }

    pub(crate) fn register(
        &mut self,
        device_id: DeviceId,
        num_devices: u32,
        global_params: Option<GlobalRegisterParams>,
    ) -> RegisterResult {
        let (callback, rec) = std::sync::mpsc::sync_channel::<RegisterResult>(1);

        self.channel
            .send(Message::Register {
                device_id,
                num_devices,
                global_params,
                callback,
            })
            .unwrap();

        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }

    pub(crate) fn all_reduce(
        &self,
        device_id: DeviceId,
        tensor: B::FloatTensorPrimitive,
        params: &SharedAllReduceParams,
    ) -> AllReduceResult<B::FloatTensorPrimitive> {
        let (callback, rec) =
            std::sync::mpsc::sync_channel::<AllReduceResult<B::FloatTensorPrimitive>>(1);

        self.channel
            .send(Message::AllReduce {
                device_id,
                tensor,
                params: params.clone(),
                callback,
            })
            .unwrap();

        // returns a tensor primitive that may or may not be on the correct device,
        // depending on the strategy used.
        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }

    pub(crate) fn finish(&self, id: DeviceId) -> FinishResult {
        let (callback, rec) = std::sync::mpsc::sync_channel::<FinishResult>(1);
        self.channel.send(Message::Finish { id, callback }).unwrap();

        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }
}

impl<B: Backend> LocalCollectiveServer<B> {
    fn new(rec: Receiver<Message<B>>) -> Self {
        Self {
            message_rec: rec,
            registered_ids: vec![],
            cur_num_devices: None,
            cur_allreduce_params: None,
            tensors: vec![],
            callbacks_register: vec![],
            callbacks_all_reduce: vec![],
            global_client: None,
        }
    }

    /// Starts the local collective server thread
    pub(crate) fn start() -> LocalCollectiveClient<B> {
        let (sender, rec) = std::sync::mpsc::sync_channel::<Message<B>>(50);

        let runtime = get_server_runtime();

        runtime.spawn(async {
            let typeid = core::any::TypeId::of::<B>();
            eprintln!("Starting server for backend: {typeid:?}");
            let mut aggregator = LocalCollectiveServer::new(rec);

            loop {
                match aggregator.message_rec.recv() {
                    Ok(message) => aggregator.process_message(message).await,
                    Err(err) => {
                        log::error!(
                            "Error receiving message from local collective server: {err:?}"
                        );
                        break;
                    }
                }
            }
        });

        LocalCollectiveClient { channel: sender }
    }

    async fn process_message(&mut self, message: Message<B>) {
        match message {
            Message::AllReduce {
                device_id,
                tensor,
                params,
                callback,
            } => {
                self.process_all_reduce_message(device_id, tensor, params, callback)
                    .await
            }
            Message::Register {
                device_id,
                num_devices,
                global_params: global,
                callback,
            } => {
                self.process_register_message(device_id, num_devices, global, callback)
                    .await
            }
            Message::Reset => self.reset(),
            Message::Finish { id, callback } => self.process_finish_message(id, callback).await,
        }
    }

    fn reset(&mut self) {
        self.registered_ids.clear();
        self.tensors.clear();
        self.cur_allreduce_params = None;
    }

    async fn process_finish_message(&mut self, id: DeviceId, callback: SyncSender<RegisterResult>) {
        if !self.registered_ids.contains(&id) {
            callback
                .send(Err(CollectiveError::MultipleUnregister))
                .unwrap();
            return;
        }

        // Remove registered with id
        self.registered_ids.retain(|x| *x != id);

        if self.registered_ids.is_empty() {
            if let Some(global_client) = self.global_client.as_mut() {
                global_client.finish().await;
            }
        }

        callback.send(Ok(())).unwrap();
    }

    async fn process_all_reduce_message(
        &mut self,
        device_id: DeviceId,
        tensor: <B as Backend>::FloatTensorPrimitive,
        params: SharedAllReduceParams,
        callback: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
    ) {
        if !self.registered_ids.contains(&device_id) {
            callback
                .send(Err(CollectiveError::RegisterNotFirstOperation))
                .unwrap();
            return;
        }

        if self.tensors.is_empty() || self.cur_allreduce_params.is_none() {
            self.cur_allreduce_params = Some(params);
        } else if self.cur_allreduce_params.clone().unwrap() != params {
            callback
                .send(Err(CollectiveError::AllReduceParamsMismatch))
                .unwrap();
            return;
        }

        self.tensors.push(tensor);
        self.callbacks_all_reduce.push(callback);

        let tensor_count = self.tensors.len();
        if tensor_count > 0 && tensor_count == self.registered_ids.len() {
            // all registered callers have sent a tensor to aggregate
            self.all_reduce().await;
        }
    }

    async fn process_register_message(
        &mut self,
        device_id: DeviceId,
        num_devices: u32,
        global_params: Option<GlobalRegisterParams>,
        callback: SyncSender<RegisterResult>,
    ) {
        if self.registered_ids.contains(&device_id) {
            let result = Err(CollectiveError::MultipleRegister);
            callback.send(result).unwrap();
            return;
        }
        if self.registered_ids.is_empty() || self.cur_num_devices.is_none() {
            self.cur_num_devices = Some(num_devices);
        } else if self.cur_num_devices.unwrap() != num_devices {
            let result = Err(CollectiveError::RegisterParamsMismatch);
            callback.send(result).unwrap();
            return;
        }

        self.registered_ids.push(device_id);
        self.callbacks_register.push(callback.clone());

        if let Some(global_params) = &global_params {
            if self.global_client.is_none() {
                let server = WsServer::new(global_params.client_data_port);
                let client = GlobalCollectiveClient::new(
                    &global_params.server_address,
                    &global_params.client_address,
                    server,
                );
                self.global_client = Some(client)
            }
        }

        // All have registered, callback
        if self.registered_ids.len() == num_devices as usize {
            if let Some(global_params) = global_params {
                let client = self
                    .global_client
                    .as_mut()
                    .expect("Global client should be initialized");

                let res = client.register(num_devices, global_params).await;

                if let Err(err) = res {
                    callback.send(Err(CollectiveError::Global(err))).unwrap();
                    return;
                }
            }

            for callback in self.callbacks_register.drain(..) {
                callback.send(Ok(())).unwrap();
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
            let tensor = outs.remove(0);
            let params = global_strategy.as_ref().unwrap();

            let device = B::float_device(&tensor);
            let tensor = global_client
                .all_reduce(tensor, *params, &device, *kind)
                .await;

            match tensor {
                Err(err) => {
                    self.send_err_all_reduce_callbacks(CollectiveError::Global(err));
                    return;
                }
                Ok(tensor) => {
                    // replace all results with global aggregation result
                    let len = outs.len() + 1;
                    outs = vec![tensor; len];
                }
            }
        }

        // Callbacks return results
        for callback in self.callbacks_all_reduce.drain(..) {
            let result = Ok(outs.remove(0));
            callback.send(result).unwrap();
        }
    }

    /// Send an error to all the subscrived all-reduce callbacks, and drain the list
    fn send_err_all_reduce_callbacks(&mut self, err: CollectiveError) {
        for callback in self.callbacks_all_reduce.drain(..) {
            let result = Err(err.clone());
            callback.send(result).unwrap();
        }
    }
}
