use std::sync::{Arc, Condvar, Mutex};

use burn_backend::ops::TensorRef;
use burn_backend::{Backend, ModuleParamId, ShardedParams};

use crate::NodeId;
use crate::collections::HashMap;

use crate::grad_sync::client::GradientSyncMessage;

#[derive(new, Debug)]
struct SyncParams {
    sharded_params: ShardedParams,
    n_required: usize,
}

pub(crate) struct GradientSyncServer<B: Backend> {
    nodes_sync_parameters: HashMap<NodeId, SyncParams>,
    all_reduce_ops_queue: HashMap<ModuleParamId, Vec<TensorRef<B>>>,
    param_required_map: HashMap<ModuleParamId, usize>,
    num_devices: usize,
    devices_registered: usize,
    is_finished_fence: Arc<(Mutex<bool>, Condvar)>,
}

impl<B: Backend> GradientSyncServer<B> {
    /// Create a new gradient syncer serveer instance.
    pub(crate) fn new(num_devices: usize, is_finished_fence: Arc<(Mutex<bool>, Condvar)>) -> Self {
        Self {
            nodes_sync_parameters: HashMap::default(),
            all_reduce_ops_queue: HashMap::default(),
            param_required_map: HashMap::default(),
            num_devices,
            devices_registered: 0,
            is_finished_fence,
        }
    }

    /// Process message from client.
    pub(crate) fn process_message(&mut self, msg: GradientSyncMessage<B>) {
        match msg {
            GradientSyncMessage::RegisterDevice((n_required_map, sharded_params_map)) => {
                self.register_device(n_required_map, sharded_params_map)
            }
            GradientSyncMessage::Register((id, tensor)) => self.on_register(id, tensor),
        }
    }

    /// Called at the start of the backward process. Lets the device announce what parameters are nodes in the autodiff graph and how many times they are required.
    fn register_device(
        &mut self,
        n_required_map: HashMap<NodeId, usize>,
        sharded_params_map: HashMap<NodeId, ShardedParams>,
    ) {
        let (lock, _) = &*self.is_finished_fence;
        let mut finished = lock.lock().unwrap();
        *finished = false;

        sharded_params_map.iter().for_each(|(k, v)| {
            self.nodes_sync_parameters.insert(
                *k,
                SyncParams::new(v.clone(), *n_required_map.get(k).unwrap_or(&1)),
            );
            let param_id = v
                .param_id
                .expect("Sharded tensor should have a parameter ID.");
            *self.param_required_map.entry(param_id).or_insert(0) += 1;
        });
        self.devices_registered += 1;

        // println!("nodes_sync_params : {:?}", self.nodes_sync_parameters);
        // println!("nodes_sync_params : {:?}", self.param_required_map);
    }

    /// Called on registration of a gradient. Calls the all_reduce operation for any parameter that is no longer required in the autodiff graph.
    fn on_register(&mut self, id: NodeId, tensor: TensorRef<B>) {
        if let Some(sync_params) = self.nodes_sync_parameters.get_mut(&id) {
            sync_params.n_required -= 1;
            if sync_params.n_required == 0 {
                let param_id = &sync_params
                    .sharded_params
                    .param_id
                    .expect("Sharded tensor should have a parameter ID.");
                let all_reduce_ops_queue =
                    self.all_reduce_ops_queue.entry(*param_id).or_insert(vec![]);
                all_reduce_ops_queue.push(tensor);

                // println!("queuing : {:?}", param_id);
            }
        }

        if self.devices_registered == self.num_devices {
            for (param_id, num_tensors) in self.param_required_map.clone() {
                let all_reduce_ops_queue =
                    self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);

                if num_tensors == all_reduce_ops_queue.len() {
                    // println!("execute : {:?}", param_id);
                    unsafe {
                        B::all_reduce_inplace(
                            all_reduce_ops_queue.to_vec(),
                            burn_backend::AllReduceStrategy::Centralized,
                            burn_backend::ReduceOperation::Mean,
                        );
                    }
                    self.all_reduce_ops_queue.remove(&param_id).unwrap();
                    self.param_required_map.remove(&param_id).unwrap();
                }
            }
        }
    }

    pub(crate) fn is_finished(&mut self) -> bool {
        if self.param_required_map.is_empty() {
            if !self.all_reduce_ops_queue.is_empty() {
                log::warn!("All reduce operations left hanging.")
            }
            self.devices_registered = 0;
            true
        } else {
            false
        }
    }
}

// TODO: Tests
