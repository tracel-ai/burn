use crate::{
    CollectiveConfig, CollectiveError, PeerId, ReduceOperation,
    global::node::base::Node,
    local::{
        AllReduceOp, AllReduceResult, BroadcastOp, BroadcastResult, ReduceOp, ReduceResult, client::LocalCollectiveClient,
    },
};
use burn_communication::websocket::{WebSocket, WsServer};
use burn_tensor::{TensorMetadata, backend::Backend};
use std::sync::{MutexGuard, OnceLock};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::Debug,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, SyncSender},
    },
};
use tokio::runtime::{Builder, Runtime};

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

    /// The collective configuration. Must be the same by every peer when calling register
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
    /// Register a new peer with the collective.
    Register {
        device_id: PeerId,
        device: B::Device,
        config: CollectiveConfig,
        callback: SyncSender<RegisterResult>,
    },
    /// Perform an all-reduce operation.
    AllReduce {
        device_id: PeerId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
        callback: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
    },
    /// Perform a reduce operation.
    Reduce {
        device_id: PeerId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
        root: PeerId,
        callback: SyncSender<ReduceResult<B::FloatTensorPrimitive>>,
    },
    /// Perform a broadcast operation (one-sender, many-receiver).
    Broadcast {
        device_id: PeerId,
        tensor: Option<B::FloatTensorPrimitive>,
        callback: SyncSender<BroadcastResult<B::FloatTensorPrimitive>>,
    },
    /// Reset the collective server.
    Reset,
    Finish {
        id: PeerId,
        callback: SyncSender<FinishResult>,
    },
}

/// The type-erased box type for [`LocalCollectiveClient<B>`].
type LocalClientBox = Box<dyn Any + Send + Sync>;

/// Global state map from [`Backend`] to boxed [`LocalCollectiveClient<B>`].
static BACKEND_CLIENT_MAP: OnceLock<Mutex<HashMap<TypeId, LocalClientBox>>> = OnceLock::new();

/// Gets a locked mutable view of the `STATE_MAP`.
pub(crate) fn get_backend_client_map() -> MutexGuard<'static, HashMap<TypeId, LocalClientBox>> {
    BACKEND_CLIENT_MAP
        .get_or_init(Default::default)
        .lock()
        .unwrap()
}

/// Get a [`LocalCollectiveClient`] for the given [`Backend`].
///
/// Will start the local collective client/server pair if necessary.
pub(crate) fn get_collective_client<B: Backend>() -> LocalCollectiveClient<B> {
    let typeid = TypeId::of::<B>();
    let mut state_map = get_backend_client_map();
    match state_map.get(&typeid) {
        Some(val) => val.downcast_ref().cloned().unwrap(),
        None => {
            let client = LocalCollectiveServer::<B>::setup(LocalCollectiveClientConfig::default());
            state_map.insert(typeid, Box::new(client.clone()));
            client
        }
    }
}

/// Global runtime.
static SERVER_RUNTIME: OnceLock<Arc<Runtime>> = OnceLock::new();

/// Get the global [`Runtime`].
pub(crate) fn get_collective_server_runtime() -> Arc<Runtime> {
    SERVER_RUNTIME
        .get_or_init(|| {
            Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("Unable to initialize runtime")
                .into()
        })
        .clone()
}

/// Configuration for the local collective client/server pair.
pub struct LocalCollectiveClientConfig {
    /// Channel capacity for the messaging queue from client to server.
    pub channel_capacity: usize,
}

impl Default for LocalCollectiveClientConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 50,
        }
    }
}

impl From<usize> for LocalCollectiveClientConfig {
    fn from(capacity: usize) -> Self {
        Self {
            channel_capacity: capacity,
        }
    }
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

    /// Setup a client/server pair with the given config.
    pub(crate) fn setup<C>(cfg: C) -> LocalCollectiveClient<B>
    where
        C: Into<LocalCollectiveClientConfig>,
    {
        let cfg = cfg.into();
        let (tx, rx) = std::sync::mpsc::sync_channel(cfg.channel_capacity);

        get_collective_server_runtime().spawn(async {
            let typeid = TypeId::of::<B>();
            log::info!("Starting server for backend: {typeid:?}");
            let mut server = LocalCollectiveServer::new(rx);

            loop {
                match server.message_rec.recv() {
                    Ok(message) => server.process_message(message).await,
                    Err(err) => {
                        log::error!(
                            "Error receiving message from local collective server: {err:?}"
                        );
                        break;
                    }
                }
            }
        });

        LocalCollectiveClient { channel: tx }
    }

    async fn process_message(&mut self, message: Message<B>) {
        match message {
            Message::Register {
                device_id,
                device,
                config,
                callback,
            } => {
                self.process_register_message(device_id, device, config, &callback)
                    .await
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

    async fn process_register_message(
        &mut self,
        device_id: PeerId,
        device: B::Device,
        config: CollectiveConfig,
        callback: &SyncSender<RegisterResult>,
    ) {
        if !config.is_valid() {
            callback.send(Err(CollectiveError::InvalidConfig)).unwrap();
            return;
        }
        if self.peers.contains(&device_id) {
            callback
                .send(Err(CollectiveError::MultipleRegister))
                .unwrap();
            return;
        }
        if self.peers.is_empty() || self.config.is_none() {
            self.config = Some(config);
        } else if *self.config.as_ref().unwrap() != config {
            callback
                .send(Err(CollectiveError::RegisterParamsMismatch))
                .unwrap();
            return;
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

            // Send results to all callbacks.
            self.callbacks_register
                .drain(..)
                .for_each(|tx| tx.send(register_result.clone()).unwrap());
        }
    }

    /// Processes an Message::AllReduce.
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

    /// Processes a Message::Reduce.
    async fn process_reduce_message(
        &mut self,
        peer_id: PeerId,
        tensor: <B as Backend>::FloatTensorPrimitive,
        op: ReduceOperation,
        root: PeerId,
        callback: SyncSender<ReduceResult<B::FloatTensorPrimitive>>,
    ) {
        if !self.peers.contains(&root) {
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

    /// Processes a Message::Broadcast.
    async fn process_broadcast_message(
        &mut self,
        caller: PeerId,
        tensor: Option<<B as Backend>::FloatTensorPrimitive>,
        callback: SyncSender<BroadcastResult<B::FloatTensorPrimitive>>,
    ) {
        if self.config.is_none() {
            callback
                .send(Err(CollectiveError::RegisterNotFirstOperation))
                .unwrap();
            return;
        }
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

    /// Reinitializes the collective server
    fn reset(&mut self) {
        self.peers.clear();
        self.all_reduce_op = None;
        self.reduce_op = None;
        self.broadcast_op = None;
    }

    /// Processes a Message::Finish.
    async fn process_finish_message(&mut self, id: PeerId, callback: SyncSender<RegisterResult>) {
        if self.config.is_none() {
            callback
                .send(Err(CollectiveError::RegisterNotFirstOperation))
                .unwrap();
            return;
        }
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
}
