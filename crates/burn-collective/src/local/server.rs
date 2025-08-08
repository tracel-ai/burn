use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{
        Arc, Mutex,
        mpsc::{Receiver, SyncSender},
    },
};

use burn_communication::websocket::{WebSocket, WsServer};
use burn_tensor::{ElementConversion, backend::Backend};
use tokio::runtime::{Builder, Runtime};

use crate::{
    AllReduceStrategy, BroadcastStrategy, CollectiveConfig, CollectiveError, PeerId,
    ReduceOperation, ReduceStrategy,
    global::node::base::Node,
    local::{
        centralized::{all_reduce_sum_centralized, broadcast_centralized, reduce_sum_centralized},
        client::LocalCollectiveClient,
        ring::all_reduce_sum_ring,
        tree::{all_reduce_sum_tree, broadcast_tree, reduce_sum_tree},
    },
};

/// Define the client/server communication on the network
type Network = WebSocket;
/// Type sent to the collective client upon completion of a register request
pub(crate) type RegisterResult = Result<(), CollectiveError>;
/// Type sent to the collective client upon completion of a all-reduce aggregation
pub(crate) type AllReduceResult<T> = Result<T, CollectiveError>;
/// Type sent to the collective client upon completion of a reduce aggregation
pub(crate) type ReduceResult<T> = Result<Option<T>, CollectiveError>;
/// Type sent to the collective client upon completion of a broadcast op
pub(crate) type BroadcastResult<T> = Result<T, CollectiveError>;
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

    /// Uncompleted all-reduce calls, one for each calling device
    all_reduce_ops: Vec<AllReduceOp<B>>,
    /// The reduce operation of the current all-reduce, as defined by the first caller
    cur_all_reduce_op: Option<ReduceOperation>,

    /// Uncompleted reduce calls, one per calling device
    reduce_ops: Vec<ReduceOp<B>>,
    /// The root peer of the current broadcast, as defined by the first caller
    cur_reduce_root: Option<PeerId>,
    /// The reduce operation of the current reduce, as defined by the first caller
    cur_reduce_op: Option<ReduceOperation>,

    /// Uncompleted broadcast calls, one for each calling device.
    broadcast_ops: Vec<BroadcastOp<B>>,
    /// The tensor passed as input for a broadcast
    cur_broadcast_input: Option<B::FloatTensorPrimitive>,
    /// The root peer of the current broadcast, as defined by the first caller
    cur_broadcast_root: Option<PeerId>,

    /// Client for global collective operations
    global_client: Option<Node<B, Network>>,
}

/// Struct for each device that calls an all-reduce operation
struct AllReduceOp<B: Backend> {
    /// Id of the caller for this operation
    caller: PeerId,
    /// The tensor primitive passed as input
    input: B::FloatTensorPrimitive,
    /// Callback for the result of the all-reduce
    result_sender: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
}

/// Struct for each device that calls an reduce operation
struct ReduceOp<B: Backend> {
    /// Id of the caller of the operation
    caller: PeerId,
    /// The tensor primitive passed as input
    input: B::FloatTensorPrimitive,
    /// Callback for the result of the reduce
    result_sender: SyncSender<ReduceResult<B::FloatTensorPrimitive>>,
}

/// Struct for each device that calls an broadcast operation
struct BroadcastOp<B: Backend> {
    /// Id of the caller of the operation
    caller: PeerId,
    /// Callback for the broadcast result
    result_sender: SyncSender<BroadcastResult<B::FloatTensorPrimitive>>,
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
            all_reduce_ops: vec![],
            cur_all_reduce_op: None,
            reduce_ops: vec![],
            cur_reduce_root: None,
            cur_reduce_op: None,
            broadcast_ops: vec![],
            cur_broadcast_input: None,
            cur_broadcast_root: None,
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
        device_id: PeerId,
        tensor: <B as Backend>::FloatTensorPrimitive,
        op: ReduceOperation,
        callback: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
    ) {
        if !self.peers.contains(&device_id) {
            callback
                .send(Err(CollectiveError::RegisterNotFirstOperation))
                .unwrap();
            return;
        }

        if self.all_reduce_ops.is_empty() || self.cur_all_reduce_op.is_none() {
            self.cur_all_reduce_op = Some(op);
        } else if self.cur_all_reduce_op.unwrap() != op {
            callback
                .send(Err(CollectiveError::AllReduceParamsMismatch))
                .unwrap();
            return;
        }

        self.all_reduce_ops.push(AllReduceOp {
            caller: device_id,
            input: tensor,
            result_sender: callback,
        });

        let tensor_count = self.all_reduce_ops.len();
        if tensor_count > 0 && tensor_count == self.peers.len() {
            // all registered callers have sent a tensor to aggregate
            let res = self.all_reduce().await;
            if let Err(err) = res {
                // Send error to all subscribers
                self.all_reduce_ops.iter_mut().for_each(|op| {
                    op.result_sender.send(Err(err.clone())).unwrap();
                });
            }
        }
    }

    /// Processes a reduce request from a client
    async fn process_reduce_message(
        &mut self,
        device_id: PeerId,
        tensor: <B as Backend>::FloatTensorPrimitive,
        op: ReduceOperation,
        root: PeerId,
        callback: SyncSender<ReduceResult<B::FloatTensorPrimitive>>,
    ) {
        if !self.peers.contains(&device_id) {
            callback
                .send(Err(CollectiveError::RegisterNotFirstOperation))
                .unwrap();
            return;
        }

        // Assign the root and reduce op, or send error if they don't match the previous
        if self.reduce_ops.is_empty() {
            self.cur_reduce_op = Some(op);
            self.cur_reduce_root = Some(root);
        } else if self.cur_reduce_op.unwrap() != op || self.cur_reduce_root.unwrap() != root {
            callback
                .send(Err(CollectiveError::ReduceParamsMismatch))
                .unwrap();
            return;
        }

        self.reduce_ops.push(ReduceOp {
            caller: device_id,
            input: tensor,
            result_sender: callback,
        });

        let tensor_count = self.reduce_ops.len();
        if tensor_count > 0 && tensor_count == self.peers.len() {
            // Do reduce
            let res = self.reduce().await;
            if let Err(err) = res {
                // Send error to all subscribers
                self.reduce_ops.iter_mut().for_each(|op| {
                    op.result_sender.send(Err(err.clone())).unwrap();
                });
            }
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

        if tensor.is_some() {
            // Assign the root, or send error if we already had a root
            if self.cur_broadcast_root.is_some() {
                callback
                    .send(Err(CollectiveError::BroadcastMultipleTensors))
                    .unwrap();
                return;
            }
            self.cur_broadcast_root = Some(caller);
            self.cur_broadcast_input = tensor;
        }

        self.broadcast_ops.push(BroadcastOp {
            caller,
            result_sender: callback,
        });

        let tensor_count = self.broadcast_ops.len();
        if tensor_count > 0 && tensor_count == self.peers.len() {
            // Do broadcast
            let res = self.broadcast().await;
            if let Err(err) = res {
                // Send error to all subscribers
                self.reduce_ops.iter_mut().for_each(|op| {
                    op.result_sender.send(Err(err.clone())).unwrap();
                });
            }
        }
    }

    /// Perform an all-reduce operation. Empties the all_reduces_ops map.
    async fn all_reduce(&mut self) -> Result<(), CollectiveError> {
        let mut tensors = HashMap::new();
        for op in &self.all_reduce_ops {
            tensors.insert(op.caller, op.input.clone());
        }

        let op = self.cur_all_reduce_op.unwrap();
        let config = self.config.as_ref().unwrap();
        if let Some(global_client) = &mut self.global_client {
            Self::all_reduce_with_global(&mut tensors, op, config, global_client).await?;
        } else {
            Self::all_reduce_local_only(&mut tensors, op, config).await?;
        }

        // Return resulting tensors
        self.all_reduce_ops.iter_mut().for_each(|op| {
            let result = tensors.remove(&op.caller).unwrap();
            op.result_sender.send(Ok(result)).unwrap();
        });

        Ok(())
    }

    /// Perform an all-reduce with no multi-node operations (global ops)
    async fn all_reduce_local_only(
        tensors: &mut HashMap<PeerId, B::FloatTensorPrimitive>,
        op: ReduceOperation,
        config: &CollectiveConfig,
    ) -> Result<(), CollectiveError> {
        let local_strategy = &config.local_all_reduce_strategy;
        match local_strategy {
            AllReduceStrategy::Centralized => all_reduce_sum_centralized::<B>(tensors),
            AllReduceStrategy::Tree(arity) => all_reduce_sum_tree::<B>(tensors, *arity),
            AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors),
        };

        if op == ReduceOperation::Mean {
            // Apply mean division
            let tensor_count = tensors.len() as f32;
            tensors.iter_mut().for_each(|(_, tensor)| {
                *tensor = B::float_div_scalar(tensor.clone(), tensor_count.elem())
            });
        }

        Ok(())
    }

    /// Do an all-reduce in a multi-node context
    ///
    /// With Tree and Centralized strategies, the all-reduce is split between a
    /// reduce (all tensors are reduced to one device), and a broadcast (the result is sent to all
    /// other devices). The all-reduce on the global level is done between both steps.
    /// Due to the nature of the Ring strategy, this separation can't be done.
    // For the Ring strategy, this isn't possible, because it is more like a
    // reduce-scatter plus an all-gather, so using a Ring strategy locally in a multi-node
    // setup may be unadvantageous.
    async fn all_reduce_with_global(
        tensors: &mut HashMap<PeerId, B::FloatTensorPrimitive>,
        op: ReduceOperation,
        config: &CollectiveConfig,
        global_client: &mut Node<B, WebSocket>,
    ) -> Result<(), CollectiveError> {
        let local_strategy = config.local_all_reduce_strategy;
        let global_strategy = config.global_all_reduce_strategy;

        // Get corresponding devices for each peer
        let devices = tensors
            .iter()
            .map(|(id, tensor)| (*id, B::float_device(tensor)))
            .collect::<HashMap<PeerId, B::Device>>();

        // For Centralized and Tree, we only need to do a reduce here, we'll do a broadcast later
        let main_device = *tensors.keys().next().unwrap();
        let mut tensors_to_reduce = core::mem::take(tensors);
        let mut main_tensor = match local_strategy {
            AllReduceStrategy::Centralized => {
                reduce_sum_centralized::<B>(tensors_to_reduce, &main_device)
            }
            AllReduceStrategy::Tree(arity) => {
                reduce_sum_tree::<B>(tensors_to_reduce, &main_device, arity)
            }
            AllReduceStrategy::Ring => {
                all_reduce_sum_ring::<B>(&mut tensors_to_reduce);
                tensors_to_reduce.remove(&main_device).unwrap()
            }
        };

        // Do aggregation on global level with the main tensor
        main_tensor = global_client
            .all_reduce(main_tensor, global_strategy.unwrap(), op)
            .await
            .map_err(CollectiveError::Global)?;

        // Broadcast result to all devices
        *tensors = match local_strategy {
            AllReduceStrategy::Tree(arity) => {
                broadcast_tree::<B>(devices, main_device, main_tensor, arity)
            }
            // If we chose the ring strategy and we must still broadcast the global result,
            // we use the centralized strategy for broadcasting, but the tree may be better.
            AllReduceStrategy::Centralized | AllReduceStrategy::Ring => {
                broadcast_centralized::<B>(devices, main_device, main_tensor)
            }
        };

        Ok(())
    }

    async fn reduce(&mut self) -> Result<(), CollectiveError> {
        let mut tensors = HashMap::new();
        for op in &self.reduce_ops {
            tensors.insert(op.caller, op.input.clone());
        }
        let tensor_count = tensors.len() as f32;

        let op = self.cur_reduce_op.unwrap();
        let config = self.config.as_ref().unwrap();

        let local_strategy = config.local_reduce_strategy;
        let root = self.cur_reduce_root.unwrap();

        // For Centralized and Tree, we only need to do a reduce here, we'll do a broadcast later
        let tensors_to_reduce = core::mem::take(&mut tensors);
        let mut result = match local_strategy {
            ReduceStrategy::Centralized => reduce_sum_centralized::<B>(tensors_to_reduce, &root),
            ReduceStrategy::Tree(arity) => reduce_sum_tree::<B>(tensors_to_reduce, &root, arity),
        };

        // Do aggregation on global level with the main tensor
        let mut result = if let Some(global_client) = &self.global_client {
            let global_strategy = config.global_reduce_strategy.unwrap();
            global_client
                .reduce(result, global_strategy, root, op)
                .await
                .map_err(CollectiveError::Global)?
        } else {
            // Mean division locally
            if op == ReduceOperation::Mean {
                result = B::float_div_scalar(result, tensor_count.elem())
            }
            Some(result)
        };

        // Return resulting tensor to root, None to others
        self.reduce_ops.iter_mut().for_each(|op| {
            let msg = if op.caller == root {
                Ok(result.take())
            } else {
                Ok(None)
            };
            op.result_sender.send(msg).unwrap();
        });

        Ok(())
    }

    async fn broadcast(&mut self) -> Result<(), CollectiveError> {
        let root = self.cur_broadcast_root.unwrap();
        let config = self.config.as_ref().unwrap();

        let local_strategy = config.local_broadcast_strategy;
        let mut root_tensor = self.cur_broadcast_input.take();

        // Get corresponding devices for each peer
        let devices = self
            .broadcast_ops
            .iter()
            .map(|op| {
                let device = self.devices.get(&op.caller).unwrap().clone();
                (op.caller, device)
            })
            .collect::<HashMap<PeerId, B::Device>>();

        // Do aggregation on global level with the main tensor
        if let Some(global_client) = &self.global_client {
            let global_strategy = config.global_broadcast_strategy.unwrap();
            let global_result = global_client
                .broadcast(root_tensor.clone(), global_strategy, root)
                .await
                .map_err(CollectiveError::Global)?;
            root_tensor = Some(global_result)
        }

        // At this point tensor must be defined
        let Some(root_tensor) = root_tensor else {
            return Err(CollectiveError::BroadcastNoTensor);
        };

        // Broadcast locally
        let mut results = match local_strategy {
            BroadcastStrategy::Tree(arity) => {
                broadcast_tree::<B>(devices, root, root_tensor, arity)
            }
            BroadcastStrategy::Centralized => {
                broadcast_centralized::<B>(devices, root, root_tensor)
            }
        };

        // Return broadcast results
        self.broadcast_ops.iter_mut().for_each(|op| {
            let result = results.remove(&op.caller).unwrap();
            op.result_sender.send(Ok(result)).unwrap();
        });

        Ok(())
    }

    // Reinitializes the collective server
    fn reset(&mut self) {
        self.peers.clear();
        self.reset_all_reduce_op();
        self.reset_reduce_op();
        self.reset_broadcast_op();
    }

    /// Resets the temporary states of any ongoing all-reduce operation
    fn reset_all_reduce_op(&mut self) {
        self.all_reduce_ops.clear();
        self.cur_all_reduce_op = None;
    }

    /// Resets the temporary states of any ongoing reduce operation
    fn reset_reduce_op(&mut self) {
        self.reduce_ops.clear();
        self.cur_reduce_op = None;
        self.cur_reduce_root = None;
    }

    /// Resets the temporary states of any ongoing broadcast operation
    fn reset_broadcast_op(&mut self) {
        self.broadcast_ops.clear();
        self.cur_broadcast_root = None;
        self.cur_broadcast_input = None;
    }
}
