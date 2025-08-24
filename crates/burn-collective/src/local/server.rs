use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, SyncSender},
    },
};

use burn_communication::websocket::{WebSocket, WsServer};
use burn_tensor::{TensorMetadata, backend::Backend};
use tokio::runtime::{Builder, Runtime};

use crate::{
    CollectiveConfig, CollectiveError, PeerId, ReduceOperation,
    global::node::base::Node,
    local::{
        AllReduceOp, AllReduceResult, BroadcastOp, BroadcastResult, ReduceOp, ReduceResult,
        client::LocalCollectiveClient,
    },
};

/// Define the client/server communication on the network
type Network = WebSocket;
/// Type sent to the collective client upon completion of a register request
pub(crate) type RegisterResult = Result<(), CollectiveError>;
/// Type sent to the collective client upon completion of a finish request
pub(crate) type FinishResult = Result<(), CollectiveError>;

/// The local collective server that manages all the collective aggregation operations
/// (like all-reduce) between local threads.
/// This thread takes in messages from different clients. The clients must register, than they can
/// send an aggregate message. They must all use the same parameters for the same aggregate
/// operation.
pub(crate) struct LocalCollectiveServer<B: Backend> {
    /// Channel receiver for messages from clients
    message_rec: Receiver<Message<B>>,

    /// The collective configuration, must be the same by every peer when calling register
    config: Option<CollectiveConfig>,
    /// The ids passed to each register so far
    peers: Vec<PeerId>,
    /// Callbacks for when all registers are done
    callbacks_register: Vec<SyncSender<RegisterResult>>,
    /// Map of each peer's id and its device
    devices: HashMap<PeerId, B::Device>,

    /// Current uncompleted all-reduce operation
    all_reduce_op: Option<AllReduceOp<B>>,

    /// Current uncompleted reduce call
    reduce_op: Option<ReduceOp<B>>,

    /// Uncompleted broadcast calls, one for each calling device.
    broadcast_op: Option<BroadcastOp<B>>,

    /// Client for global collective operations
    global_client: Option<Node<B, Network>>,
}

#[derive(Debug)]
pub(crate) enum Message<B: Backend> {
    Register {
        device_id: PeerId,
        device: B::Device,
        config: CollectiveConfig,
        callback: SyncSender<RegisterResult>,
    },
    AllReduce {
        device_id: PeerId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
        callback: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
    },
    Reduce {
        device_id: PeerId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
        root: PeerId,
        callback: SyncSender<ReduceResult<B::FloatTensorPrimitive>>,
    },
    Broadcast {
        device_id: PeerId,
        tensor: Option<B::FloatTensorPrimitive>,
        callback: SyncSender<BroadcastResult<B::FloatTensorPrimitive>>,
    },
    Reset,
    Finish {
        id: PeerId,
        callback: SyncSender<FinishResult>,
    },
}

// HashMap for each server by Backend type
static STATE: Mutex<Option<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = Mutex::new(None);

/// Get a client for the local collecive server, starting the server if necessary
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

impl<B: Backend> LocalCollectiveServer<B> {
    fn new(rec: Receiver<Message<B>>) -> Self {
        Self {
            message_rec: rec,
            config: None,
            peers: vec![],
            devices: HashMap::new(),
            all_reduce_op: None,
            reduce_op: None,
            broadcast_op: None,
            callbacks_register: vec![],
            global_client: None,
        }
    }

    /// Starts the local collective server thread
    pub(crate) fn start() -> LocalCollectiveClient<B> {
        let (sender, rec) = std::sync::mpsc::sync_channel::<Message<B>>(50);

        let runtime = get_server_runtime();

        runtime.spawn(async {
            let typeid = core::any::TypeId::of::<B>();
            log::info!("Starting server for backend: {typeid:?}");
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
            Message::Register {
                device_id,
                device,
                config,
                callback,
            } => {
                if let Err(err) = self
                    .process_register_message(device_id, device, config, &callback)
                    .await
                {
                    callback.send(Err(err)).unwrap()
                }
            }
            Message::AllReduce {
                device_id,
                tensor,
                op,
                callback,
            } => {
                self.process_all_reduce_message(device_id, tensor, op, callback)
                    .await
            }
            Message::Reduce {
                device_id,
                tensor,
                op,
                root,
                callback,
            } => {
                self.process_reduce_message(device_id, tensor, op, root, callback)
                    .await
            }
            Message::Broadcast {
                device_id,
                tensor,
                callback,
            } => {
                self.process_broadcast_message(device_id, tensor, callback)
                    .await
            }
            Message::Reset => self.reset(),
            Message::Finish { id, callback } => self.process_finish_message(id, callback).await,
        }
    }

    async fn process_finish_message(&mut self, id: PeerId, callback: SyncSender<RegisterResult>) {
        if !self.peers.contains(&id) {
            callback
                .send(Err(CollectiveError::MultipleUnregister))
                .unwrap();
            return;
        }

        // Remove registered with id
        self.peers.retain(|x| *x != id);

        if self.peers.is_empty()
            && let Some(mut global_client) = self.global_client.take()
        {
            global_client.finish().await;
        }

        callback.send(Ok(())).unwrap();
    }

    async fn process_register_message(
        &mut self,
        device_id: PeerId,
        device: B::Device,
        config: CollectiveConfig,
        callback: &SyncSender<RegisterResult>,
    ) -> Result<(), CollectiveError> {
        if !config.is_valid() {
            return Err(CollectiveError::InvalidConfig);
        }
        if self.peers.contains(&device_id) {
            return Err(CollectiveError::MultipleRegister);
        }
        if self.peers.is_empty() || self.config.is_none() {
            self.config = Some(config);
        } else if *self.config.as_ref().unwrap() != config {
            return Err(CollectiveError::RegisterParamsMismatch);
        }

        self.peers.push(device_id);
        self.callbacks_register.push(callback.clone());
        self.devices.insert(device_id, device);

        let config = self.config.as_ref().unwrap();
        let global_params = config.global_register_params();
        if let Some(global_params) = &global_params
            && self.global_client.is_none()
        {
            let server = WsServer::new(global_params.data_service_port);
            let client = Node::new(&global_params.global_address, server);
            self.global_client = Some(client)
        }

        // All have registered, callback
        if self.peers.len() == config.num_devices {
            let mut register_result = Ok(());

            // if an error occurs on the global register, it must be passed back to every local peer
            if let Some(global_params) = global_params {
                let client = self
                    .global_client
                    .as_mut()
                    .expect("Global client should be initialized");

                register_result = client
                    .register(self.peers.clone(), global_params)
                    .await
                    .map_err(CollectiveError::Global);
            };

            for callback in self.callbacks_register.drain(..) {
                callback.send(register_result.clone()).unwrap();
            }
        }

        Ok(())
    }

    /// Processes an all-reduce request from a client, registers the input tensor and output
    /// callback
    async fn process_all_reduce_message(
        &mut self,
        peer_id: PeerId,
        tensor: <B as Backend>::FloatTensorPrimitive,
        op: ReduceOperation,
        callback: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
    ) {
        if !self.peers.contains(&peer_id) {
            callback
                .send(Err(CollectiveError::RegisterNotFirstOperation))
                .unwrap();
            return;
        }

        if self.all_reduce_op.is_none() {
            // First call to all-reduce
            self.all_reduce_op = Some(AllReduceOp::new(tensor.shape(), op));
        }
        // Take the operation, we'll put it back if we're not done
        let mut all_reduce_op = self.all_reduce_op.take().unwrap();

        // On the last caller, the all-reduce is done here
        let res =
            all_reduce_op.register_call(peer_id, tensor, callback.clone(), op, self.peers.len());

        // Upon an error or the last call, the all_reduce_op is dropped
        match res {
            Ok(is_ready) => {
                if is_ready {
                    all_reduce_op
                        .execute(self.config.as_ref().unwrap(), &mut self.global_client)
                        .await;
                } else {
                    // Put operation back, we're waiting for more calls
                    self.all_reduce_op = Some(all_reduce_op)
                }
            }
            Err(err) => all_reduce_op.send_err_to_all(err),
        }
    }

    /// Processes a reduce request from a client
    async fn process_reduce_message(
        &mut self,
        peer_id: PeerId,
        tensor: <B as Backend>::FloatTensorPrimitive,
        op: ReduceOperation,
        root: PeerId,
        callback: SyncSender<ReduceResult<B::FloatTensorPrimitive>>,
    ) {
        if !self.peers.contains(&peer_id) {
            callback
                .send(Err(CollectiveError::RegisterNotFirstOperation))
                .unwrap();
            return;
        }

        if self.reduce_op.is_none() {
            // First call to reduce
            self.reduce_op = Some(ReduceOp::new(tensor.shape(), op, root));
        }
        let mut reduce_op = self.reduce_op.take().unwrap();

        // On the last caller, the all-reduce is done here
        let res = reduce_op.register_call(
            peer_id,
            tensor,
            callback.clone(),
            op,
            root,
            self.peers.len(),
        );

        // Upon an error or the last call, the all_reduce_op is dropped
        match res {
            Ok(is_ready) => {
                if is_ready {
                    reduce_op
                        .execute(root, self.config.as_ref().unwrap(), &mut self.global_client)
                        .await;
                } else {
                    // Put operation back, we're waiting for more calls
                    self.reduce_op = Some(reduce_op)
                }
            }
            Err(err) => reduce_op.send_err_to_all(err),
        }
    }

    /// Processes a broadcast request from a client
    async fn process_broadcast_message(
        &mut self,
        caller: PeerId,
        tensor: Option<<B as Backend>::FloatTensorPrimitive>,
        callback: SyncSender<BroadcastResult<B::FloatTensorPrimitive>>,
    ) {
        if !self.peers.contains(&caller) {
            callback
                .send(Err(CollectiveError::RegisterNotFirstOperation))
                .unwrap();
            return;
        }

        if self.broadcast_op.is_none() {
            // First call to broadcast
            self.broadcast_op = Some(BroadcastOp::new());
        }
        let device = self.devices.get(&caller).unwrap().clone();

        let mut broadcast_op = self.broadcast_op.take().unwrap();

        // On the last caller, the all-reduce is done here
        let res =
            broadcast_op.register_call(caller, tensor, callback.clone(), device, self.peers.len());

        // Upon an error or the last call, the all_reduce_op is dropped
        match res {
            Ok(is_ready) => {
                if is_ready {
                    broadcast_op
                        .execute(self.config.as_ref().unwrap(), &mut self.global_client)
                        .await;
                } else {
                    // Put operation back, we're waiting for more calls
                    self.broadcast_op = Some(broadcast_op)
                }
            }
            Err(err) => broadcast_op.send_err_to_all(err),
        }
    }

    // Reinitializes the collective server
    fn reset(&mut self) {
        self.peers.clear();
        self.all_reduce_op = None;
        self.reduce_op = None;
        self.broadcast_op = None;
    }
}
