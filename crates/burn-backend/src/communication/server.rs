use burn_std::tensor;
use std::collections::HashMap;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crate::all_reduce::all_reduce_inplace_sum_centralized;
use crate::client::GradientSyncMessage;
use crate::ops::TensorRef;
use crate::tensor::Device;
use crate::worker::{AllReduceArgs, CollectiveOperationMessage, Worker};
use crate::{Backend, ModuleParamId, PeerId, ReduceOperation, ShardedParams};
use crate::{DeviceOps, TensorMetadata};

pub(crate) struct GradientSyncServer<B: Backend> {
    all_reduce_ops_queue: HashMap<ModuleParamId, Vec<TensorRef<B>>>,
    param_required_map: HashMap<ModuleParamId, usize>,
    devices: HashMap<PeerId, Device<B>>,
    num_devices: usize,
    devices_registered: usize,
    syncing_devices: Vec<Device<B>>,
    devices_synced: usize,
    is_finished_fence: Arc<(Mutex<bool>, Condvar)>,
    task_senders: HashMap<PeerId, Sender<CollectiveOperationMessage<B>>>,
    sync_barriers: Vec<Arc<(Mutex<bool>, Condvar)>>,
}

impl<B: Backend> GradientSyncServer<B> {
    /// Create a new gradient sync server instance.
    pub(crate) fn new(
        devices: Vec<B::Device>,
        is_finished_fence: Arc<(Mutex<bool>, Condvar)>,
    ) -> Self {
        let mut task_senders = HashMap::default();
        let mut devices_map = HashMap::default();
        for i in 0..devices.len() {
            let (tx, rx) = std::sync::mpsc::channel();
            let peer_id = PeerId::from(i);
            task_senders.insert(peer_id, tx);
            let device = devices[i].clone();
            devices_map.insert(peer_id, device.clone());
            thread::spawn(move || {
                let worker = Worker::new(rx);
                worker.run();
            });
        }
        Self {
            all_reduce_ops_queue: HashMap::default(),
            param_required_map: HashMap::default(),
            devices: devices_map,
            num_devices: devices.len(),
            devices_registered: 0,
            syncing_devices: vec![],
            devices_synced: 0,
            is_finished_fence,
            task_senders,
            sync_barriers: vec![],
        }
    }

    /// Process message from client.
    pub(crate) fn process_message(&mut self, msg: GradientSyncMessage<B>) {
        match msg {
            GradientSyncMessage::RegisterDevice(params) => self.register_device(params),
            GradientSyncMessage::Register((tensor, params)) => self.on_register(tensor, params),
            GradientSyncMessage::Sync((device, is_synced)) => self.sync(device, is_synced),
            // {
            //     self.waiting_devices += 1;
            //     if self.waiting_devices == self.num_devices {
            //         self.update_finished(&device);
            //     }
            // }
        }
    }

    fn try_flush_sync(&mut self) {
        println!("try flush sync");
        println!("{:?}", self.all_reduce_ops_queue.len());
        println!("{:?}", self.syncing_devices);
        println!("{:?}", self.sync_barriers);
        if self.all_reduce_ops_queue.is_empty() {
            println!("empty queue");
            for (d, barrier) in self.syncing_devices.iter().zip(self.sync_barriers.clone()) {
                println!("[{:?}] comm server sync", thread::current().id());
                println!("launching sync {d:?}");
                self.task_senders
                    .get(&PeerId::from(d.id().index_id))
                    .unwrap()
                    .send(CollectiveOperationMessage::Sync(d.clone()))
                    .unwrap();
                // B::collective_sync_native(&d);
                println!("launched sync {d:?}");
                let (lock, cvar) = &*barrier;
                println!("acquired lock {d:?}");
                let mut synced = lock.lock().unwrap();
                *synced = true;
                cvar.notify_all();
                self.devices_synced += 1;
                println!("synced collective {d:?}");
            }
            self.syncing_devices.clear();
            self.sync_barriers.clear();
        }

        if self.devices_synced == self.num_devices {
            self.devices_registered = 0;
            self.syncing_devices.clear();
            self.devices_synced = 0;
            self.all_reduce_ops_queue.clear();
            self.param_required_map.clear();
            self.sync_barriers.clear();
            println!("is_finished ");
        }
    }

    fn sync(&mut self, device: Device<B>, sync_barrier: Arc<(Mutex<bool>, Condvar)>) {
        let mut is_finished = false;
        if B::supports_native_collective(&device) {
            self.syncing_devices.push(device.clone());
            self.sync_barriers.push(sync_barrier);
            self.try_flush_sync();
            println!("syncing collective {device:?}");
            is_finished = self.devices_synced == self.num_devices;
        } else {
            self.sync_barriers.push(sync_barrier);
            if self.sync_barriers.len() == self.num_devices {
                is_finished = true;
                for barrier in self.sync_barriers.clone() {
                    let (lock, cvar) = &*barrier;
                    let mut synced = lock.lock().unwrap();
                    *synced = true;
                    cvar.notify_all();
                }
            }
        }

        if is_finished {
            self.devices_registered = 0;
            self.syncing_devices.clear();
            self.devices_synced = 0;
            self.all_reduce_ops_queue.clear();
            self.param_required_map.clear();
            self.sync_barriers.clear();
            println!("is_finished {device:?}");
        }
    }

    pub(crate) fn close(&mut self, callback: Sender<()>) {
        for sender in self.task_senders.values() {
            sender.send(CollectiveOperationMessage::Close()).unwrap();
        }
        callback.send(()).unwrap();
    }

    /// Called at the start of the backward process. Lets the device announce what parameters are nodes in the autodiff graph and how many times they are required.
    fn register_device(&mut self, sharded_params: Vec<ShardedParams>) {
        let (lock, _) = &*self.is_finished_fence;
        let mut finished = lock.lock().unwrap();
        *finished = false;
        drop(finished);

        sharded_params.iter().for_each(|param| {
            let id = param
                .param_id
                .expect("Sharded parameters should have a module parameter ID.");
            *self.param_required_map.entry(id).or_insert(0) += 1;
        });
        self.devices_registered += 1;

        println!("Device registered: {}", self.devices_registered);
    }

    fn launch_ops(&mut self) {
        if self.devices_registered == self.num_devices {
            for (param_id, num_tensors) in self.param_required_map.clone() {
                let all_reduce_ops_queue =
                    self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);

                let devices: Vec<B::Device> = all_reduce_ops_queue
                    .to_vec()
                    .iter()
                    .map(|tensor| B::comm_device(tensor))
                    .collect();
                if num_tensors == all_reduce_ops_queue.len() {
                    println!("Launching for {param_id:?}");
                    if B::supports_native_collective(&devices[0]) {
                        let peer_ids: Vec<PeerId> = devices
                            .iter()
                            .map(|d| PeerId::from(d.id().index_id))
                            .collect();
                        for t in all_reduce_ops_queue.to_vec() {
                            let peer_id = PeerId::from(B::comm_device(&t).id().index_id);
                            // self.task_senders
                            //     .get(&peer_id)
                            //     .expect("Peer ID was registered.")
                            //     .send(CollectiveOperationMessage::AllReduce(AllReduceArgs {
                            //         tensor: t,
                            //         device_ids: peer_ids.clone(),
                            //     }))
                            //     .expect("Can send to worker");
                            println!("[{:?}] comm server all_reduces", thread::current().id());
                            B::all_reduce_in_place_native(
                                t,
                                peer_id,
                                peer_ids.clone(),
                                ReduceOperation::Sum, // TODO: sum hard coded.
                            );
                            println!("launched all_reduce for {peer_id:?}");
                        }
                        // if self.num_devices == self.syncing_devices {
                        //     self.update_finished(&devices[0]);
                        // }
                    } else {
                        // TODO: operation hard coded to mean.
                        all_reduce_inplace_sum_centralized(
                            all_reduce_ops_queue.to_vec(),
                            crate::ReduceOperation::Mean,
                        );
                        // if self.num_devices == self.syncing_devices {
                        //     self.update_finished(&devices[0]);
                        // }
                    }
                    self.all_reduce_ops_queue.remove(&param_id).unwrap();
                    self.param_required_map.remove(&param_id).unwrap();
                    self.try_flush_sync();
                }
            }
        }
    }

    /// Called on registration of a gradient. Calls the all_reduce operation for any parameter that is no longer required in the autodiff graph.
    fn on_register(&mut self, tensor: TensorRef<B>, sharded_params: ShardedParams) {
        let param_id = sharded_params
            .param_id
            .expect("Sharded tensor should have a parameter ID.");
        println!("Received {param_id:?} from {:?}", B::comm_device(&tensor));
        let all_reduce_ops_queue = self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);
        all_reduce_ops_queue.push(tensor.clone());
        self.launch_ops();
    }

    // pub(crate) fn update_finished(&mut self, device: &B::Device) {
    //     let mut is_finished = false;
    //     if B::supports_native_collective(device) {
    //         if self.all_reduce_ops_queue.is_empty() {
    //             for d in self.devices.values() {
    //                 B::collective_sync_native(d);
    //             }
    //             is_finished = true;
    //         }
    //     } else {
    //         if self.all_reduce_ops_queue.is_empty() {
    //             is_finished = true;
    //         }
    //     }

    //     if is_finished {
    //         self.devices_registered = 0;
    //         self.syncing_devices = 0;
    //         self.all_reduce_ops_queue.clear();
    //         self.param_required_map.clear();

    //         let (lock, cvar) = &*self.is_finished_fence;
    //         let mut finished = lock.lock().unwrap();
    //         *finished = true;
    //         cvar.notify_all();
    //     }
    // }
}

// TODO: Tests
