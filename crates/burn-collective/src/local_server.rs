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
    AllReduceStrategy, CollectiveConfig, CollectiveError, DeviceId, GlobalRegisterParams,
    ReduceKind, SharedAllReduceParams,
    centralized::{all_reduce_sum_centralized, broadcast_centralized, reduce_sum_centralized},
    global::node::base::GlobalCollectiveClient,
    ring::all_reduce_sum_ring,
    tree::{all_reduce_sum_tree, broadcast_tree, reduce_sum_tree},
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
    /// Callbacks for when all registers are done
    callbacks_register: Vec<SyncSender<RegisterResult>>,

    /// The params of the current operation, as defined by the first caller
    cur_all_reduce_params: Option<SharedAllReduceParams>,
    /// Uncompleted all-reduce calls, one for each calling device
    all_reduce_ops: HashMap<DeviceId, AllReduceOp<B>>,

    /// Client for global collective operations
    global_client: Option<GlobalCollectiveClient<B, Network>>,
}

/// Struct for each device that calls an all-reduce operation
struct AllReduceOp<B: Backend> {
    /// The tensor primitive passed as input
    input: B::FloatTensorPrimitive,
    /// Callback for the result of the all-reduce
    result_sender: SyncSender<AllReduceResult<B::FloatTensorPrimitive>>,
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

    pub(crate) fn register(&mut self, config: &CollectiveConfig) -> RegisterResult {
        if config.is_valid() {
            return Err(CollectiveError::InvalidConfig);
        }

        let device_id = config.device_id;
        let num_devices = config.num_devices;
        let global_params = config.global_register_params();
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
        tensor: B::FloatTensorPrimitive,
        config: &CollectiveConfig,
    ) -> AllReduceResult<B::FloatTensorPrimitive> {
        if config.is_valid() {
            return Err(CollectiveError::InvalidConfig);
        }

        let device_id = config.device_id;
        let all_reduce_params = config.all_reduce_params();
        let (callback, rec) =
            std::sync::mpsc::sync_channel::<AllReduceResult<B::FloatTensorPrimitive>>(1);
        let msg = Message::AllReduce {
            device_id,
            tensor,
            params: all_reduce_params,
            callback,
        };

        self.channel.send(msg).unwrap();

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
            cur_all_reduce_params: None,
            all_reduce_ops: HashMap::new(),
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
        self.all_reduce_ops.clear();
        self.cur_all_reduce_params = None;
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

    /// Processes an all-reduce request from a client, registers the input tensor and output
    /// callback
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

        if self.all_reduce_ops.is_empty() || self.cur_all_reduce_params.is_none() {
            self.cur_all_reduce_params = Some(params);
        } else if self.cur_all_reduce_params.clone().unwrap() != params {
            callback
                .send(Err(CollectiveError::AllReduceParamsMismatch))
                .unwrap();
            return;
        }

        self.all_reduce_ops.insert(
            device_id,
            AllReduceOp {
                input: tensor,
                result_sender: callback,
            },
        );

        let tensor_count = self.all_reduce_ops.len();
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
                let server = WsServer::new(global_params.data_service_port);
                let client = GlobalCollectiveClient::new(
                    &global_params.global_address,
                    &global_params.node_address,
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

    /// Perform an all-reduce operation. Empties the all_reduces_ops map.
    async fn all_reduce(&mut self) {
        let mut tensors = HashMap::new();
        let mut results = HashMap::new();
        for (dev, op) in self.all_reduce_ops.drain() {
            tensors.insert(dev, op.input);
            results.insert(dev, op.result_sender);
        }

        let params = self.cur_all_reduce_params.as_ref().unwrap();

        let res = if let Some(global_client) = &mut self.global_client {
            Self::all_reduce_with_global(&mut tensors, params, global_client).await
        } else {
            Self::all_reduce_local_only(&mut tensors, params).await
        };

        match res {
            Err(err) => {
                // Send error to all subscribers
                results.iter_mut().for_each(|(_, callback)| {
                    callback.send(Err(err.clone())).unwrap();
                });
            }
            Ok(_) => {
                // Return resulting tensors
                results.iter_mut().for_each(|(id, callback)| {
                    let result = tensors.remove(id).unwrap();
                    callback.send(Ok(result)).unwrap();
                });
            }
        };
    }

    /// Perform an all-reduce with no multi-node operations (global ops)
    async fn all_reduce_local_only(
        tensors: &mut HashMap<DeviceId, B::FloatTensorPrimitive>,
        params: &SharedAllReduceParams,
    ) -> Result<(), CollectiveError> {
        let kind = &params.kind;
        let local_strategy = &params.local_strategy;
        match local_strategy {
            AllReduceStrategy::Centralized => all_reduce_sum_centralized::<B>(tensors),
            AllReduceStrategy::Tree(arity) => all_reduce_sum_tree::<B>(tensors, *arity),
            AllReduceStrategy::Ring => all_reduce_sum_ring::<B>(tensors),
        };

        if *kind == ReduceKind::Mean {
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
        tensors: &mut HashMap<DeviceId, B::FloatTensorPrimitive>,
        params: &SharedAllReduceParams,
        global_client: &mut GlobalCollectiveClient<B, WebSocket>,
    ) -> Result<(), CollectiveError> {
        let kind = &params.kind;
        let local_strategy = &params.local_strategy;

        // Get corresponding devices for each peer
        let devices = tensors
            .iter()
            .map(|(id, tensor)| (*id, B::float_device(tensor)))
            .collect::<HashMap<DeviceId, B::Device>>();

        // For Centralized and Tree, we only need to do a reduce here, we'll do a broadcast later
        let main_device = *tensors.keys().next().unwrap();
        let mut tensors_to_reduce = core::mem::take(tensors);
        let mut main_tensor = match local_strategy {
            AllReduceStrategy::Centralized => {
                reduce_sum_centralized::<B>(tensors_to_reduce, &main_device)
            }
            AllReduceStrategy::Tree(arity) => {
                reduce_sum_tree::<B>(tensors_to_reduce, &main_device, *arity)
            }
            AllReduceStrategy::Ring => {
                all_reduce_sum_ring::<B>(&mut tensors_to_reduce);
                tensors_to_reduce.remove(&main_device).unwrap()
            }
        };

        // Do aggregation on global level with the main tensor
        let device = B::float_device(&main_tensor);
        main_tensor = global_client
            .all_reduce(main_tensor, params.global_strategy.unwrap(), &device, *kind)
            .await
            .map_err(CollectiveError::Global)?;

        // Broadcast result to all devices
        *tensors = match local_strategy {
            AllReduceStrategy::Tree(arity) => {
                broadcast_tree::<B>(devices, main_device, main_tensor, *arity)
            }
            // If we chose the ring strategy and we must still broadcast the global result,
            // we use the centralized strategy for broadcasting, but the tree may be better
            // TODO use the optimal broadcast in case of ring strategy
            AllReduceStrategy::Centralized | AllReduceStrategy::Ring => {
                broadcast_centralized::<B>(devices, main_device, main_tensor)
            }
        };

        Ok(())
    }
}
