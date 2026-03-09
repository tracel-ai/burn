use std::collections::HashMap;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use cubecl::device::DeviceId;

use crate::DeviceOps;
use crate::all_reduce::all_reduce_inplace_sum_centralized;
use crate::client::GradientSyncMessage;
use crate::ops::TensorRef;
use crate::worker::{AllReduceArgs, Worker};
use crate::{Backend, ModuleParamId, PeerId, ShardedParams};

#[derive(new, Debug)]
struct SyncParams {
    sharded_params: ShardedParams,
    n_required: usize,
}

pub(crate) struct GradientSyncServer<B: Backend> {
    // nodes_sync_parameters: HashMap<u64, SyncParams>,
    all_reduce_ops_queue: HashMap<ModuleParamId, Vec<TensorRef<B>>>,
    param_required_map: HashMap<ModuleParamId, usize>,
    sharded_params_map: HashMap<ModuleParamId, ShardedParams>,
    num_devices: usize,
    devices_registered: usize,
    is_finished_fence: Arc<(Mutex<bool>, Condvar)>,
    task_senders: HashMap<PeerId, Sender<AllReduceArgs<B>>>,
}

impl<B: Backend> GradientSyncServer<B> {
    /// Create a new gradient sync server instance.
    pub(crate) fn new(
        devices: Vec<B::Device>,
        is_finished_fence: Arc<(Mutex<bool>, Condvar)>,
    ) -> Self {
        let mut task_senders = HashMap::default();
        println!("devices: {:?}", devices);
        for i in 0..devices.len() {
            let (tx, rx) = std::sync::mpsc::channel();
            task_senders.insert(PeerId::from(i), tx);
            let device = devices[i].clone();
            let num_devices = devices.len();
            thread::spawn(move || {
                // // Fallback implementation of all_reduce for a backend uses `burn-collective`.
                // register::<B>(
                //     PeerId::from(i),
                //     device,
                //     CollectiveConfig::default()
                //         .with_num_devices(num_devices)
                //         .with_local_all_reduce_strategy(burn_backend::AllReduceStrategy::Ring),
                // )
                // .expect("Couldn't register for collective operations!");

                let worker = Worker::new(rx);
                worker.run();
            });
        }
        Self {
            // nodes_sync_parameters: HashMap::default(),
            sharded_params_map: HashMap::default(),
            all_reduce_ops_queue: HashMap::default(),
            param_required_map: HashMap::default(),
            num_devices: devices.len(),
            devices_registered: 0,
            is_finished_fence,
            task_senders,
        }
    }

    /// Process message from client.
    pub(crate) fn process_message(&mut self, msg: GradientSyncMessage<B>) {
        match msg {
            GradientSyncMessage::RegisterDevice(params) => self.register_device(params),
            // GradientSyncMessage::RegisterDevice((n_required_map, sharded_params_map)) => {
            //     self.register_device(n_required_map, sharded_params_map)
            // }
            GradientSyncMessage::Register((tensor, params)) => self.on_register(tensor, params),
        }
    }

    /// Called at the start of the backward process. Lets the device announce what parameters are nodes in the autodiff graph and how many times they are required.
    fn register_device(&mut self, sharded_params: Vec<ShardedParams>) {
        let (lock, _) = &*self.is_finished_fence;
        let mut finished = lock.lock().unwrap();
        *finished = false;

        sharded_params.iter().for_each(|param| {
            let id = param
                .param_id
                .expect("Sharded parameters should have a module parameter ID.");
            *self.param_required_map.entry(id).or_insert(0) += 1;
            // *self.sharded_params_map.entry(id).or_insert(0) += 1;
        });
        self.devices_registered += 1;
    }
    // fn register_device(
    //     &mut self,
    //     n_required_map: HashMap<u64, usize>,
    //     sharded_params_map: HashMap<u64, ShardedParams>,
    // ) {
    //     let (lock, _) = &*self.is_finished_fence;
    //     let mut finished = lock.lock().unwrap();
    //     *finished = false;

    //     sharded_params_map.iter().for_each(|(k, v)| {
    //         self.nodes_sync_parameters.insert(
    //             *k,
    //             SyncParams::new(v.clone(), *n_required_map.get(k).unwrap_or(&1)),
    //         );
    //         let param_id = v
    //             .param_id
    //             .expect("Sharded tensor should have a parameter ID.");
    //         *self.param_required_map.entry(param_id).or_insert(0) += 1;
    //     });
    //     self.devices_registered += 1;
    // }

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
                println!("devices: {:?}", devices);
                println!(
                    "devices id: {:?}",
                    devices.iter().map(|d| d.id()).collect::<Vec<DeviceId>>()
                );
                if num_tensors == all_reduce_ops_queue.len() {
                    println!("All tensors queued, execute : {:?}", param_id);
                    if B::supports_native_communication(&devices[0]) {
                        println!("Supports native comm ops");
                        let peer_ids: Vec<PeerId> = devices
                            .iter()
                            .map(|d| PeerId::from(d.id().index_id))
                            .collect();
                        for t in all_reduce_ops_queue.to_vec() {
                            let peer_id = PeerId::from(B::comm_device(&t).id().index_id);
                            println!("sending to worker : {:?}", peer_ids);
                            println!("sending to worker : {:?}", peer_id);
                            self.task_senders
                                .get(&peer_id)
                                .expect("Peer ID was registered.")
                                .send(AllReduceArgs {
                                    tensor: t,
                                    device_ids: peer_ids.clone(),
                                })
                                .expect("Can send to worker");
                        }
                    } else {
                        println!("Dont support native comms");
                        // TODO: operation hard coded to mean.
                        all_reduce_inplace_sum_centralized(
                            all_reduce_ops_queue.to_vec(),
                            crate::ReduceOperation::Mean,
                        );
                    }
                    self.all_reduce_ops_queue.remove(&param_id).unwrap();
                    self.param_required_map.remove(&param_id).unwrap();
                }
            }
        }
    }

    /// Called on registration of a gradient. Calls the all_reduce operation for any parameter that is no longer required in the autodiff graph.
    fn on_register(&mut self, tensor: TensorRef<B>, sharded_params: ShardedParams) {
        println!("On register");
        let peer_id = PeerId::from(B::comm_device(&tensor).id().index_id);
        println!("peerid : {}", peer_id);
        let param_id = sharded_params
            .param_id
            .expect("Sharded tensor should have a parameter ID.");
        let all_reduce_ops_queue = self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);
        all_reduce_ops_queue.push(tensor.clone());
        self.launch_ops();
    }
    // fn on_register(&mut self, id: u64, tensor: TensorRef<B>) {
    //     println!("On register");
    //     if let Some(sync_params) = self.nodes_sync_parameters.get_mut(&id) {
    //         sync_params.n_required -= 1;
    //         if sync_params.n_required == 0 {
    //             let param_id = &sync_params
    //                 .sharded_params
    //                 .param_id
    //                 .expect("Sharded tensor should have a parameter ID.");
    //             let all_reduce_ops_queue =
    //                 self.all_reduce_ops_queue.entry(*param_id).or_insert(vec![]);
    //             all_reduce_ops_queue.push(tensor);

    //             // println!("queuing : {:?}", param_id);
    //         }
    //     }

    //     if self.devices_registered == self.num_devices {
    //         for (param_id, num_tensors) in self.param_required_map.clone() {
    //             let all_reduce_ops_queue =
    //                 self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);

    //             if num_tensors == all_reduce_ops_queue.len() {
    //                 // println!("execute : {:?}", param_id);
    //                 let device_ids: Vec<PeerId> = all_reduce_ops_queue
    //                     .to_vec()
    //                     .iter()
    //                     .map(|tensor| PeerId::from(B::comm_device(tensor).id().index_id))
    //                     .collect();
    //                 for t in all_reduce_ops_queue.to_vec() {
    //                     let peer_id = PeerId::from(B::comm_device(&t).id().index_id);
    //                     // B::all_reduce_inplace(
    //                     //     t.clone(),
    //                     //     PeerId::from(B::comm_device(&t).id().index_id),
    //                     //     device_ids.clone(),
    //                     //     burn_backend::ReduceOperation::Sum,
    //                     // );
    //                     println!("sending to worker : {:?}", peer_id);
    //                     self.task_senders
    //                         .get(&peer_id)
    //                         .expect("Peer ID was registered.")
    //                         .send(AllReduceArgs {
    //                             tensor: t,
    //                             device_ids: device_ids.clone(),
    //                         })
    //                         .expect("Can send to worker");
    //                 }
    //                 self.all_reduce_ops_queue.remove(&param_id).unwrap();
    //                 self.param_required_map.remove(&param_id).unwrap();
    //             }
    //         }
    //     }
    // }

    pub(crate) fn is_finished(&mut self) -> bool {
        if self.param_required_map.is_empty() {
            // if !self.all_reduce_ops_queue.is_empty() {
            //     log::warn!("All reduce operations left hanging.")
            // }
            self.devices_registered = 0;
            true
        } else {
            false
        }
    }
}

// TODO: Tests
