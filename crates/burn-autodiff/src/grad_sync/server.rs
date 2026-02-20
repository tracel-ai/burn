use std::sync::mpsc::Sender;

use burn_backend::tensor::CommunicationTensor;
use burn_backend::{Backend, ModuleParamId, ShardedParams};

use crate::NodeId;
use crate::collections::HashMap;

use crate::grad_sync::client::GradientSyncMessage;

#[derive(new, Debug)]
struct SyncParams {
    sharded_params: ShardedParams,
    n_required: usize,
}

#[derive(Default)]
pub(crate) struct GradientSyncServer<B: Backend> {
    nodes_sync_parameters: HashMap<NodeId, SyncParams>,
    all_reduce_ops_queue: HashMap<ModuleParamId, Vec<CommunicationTensor<B>>>,
    param_required_map: HashMap<ModuleParamId, usize>,
}

impl<B: Backend> GradientSyncServer<B> {
    pub(crate) fn process_message(&mut self, msg: GradientSyncMessage<B>) {
        match msg {
            GradientSyncMessage::RegisterDevice((n_required_map, sharded_params_map)) => {
                self.register_device(n_required_map, sharded_params_map)
            }
            GradientSyncMessage::Register((id, tensor)) => self.on_register(id, tensor),
            GradientSyncMessage::IsFinished(sender) => self.is_finished(sender),
        }
    }

    fn register_device(
        &mut self,
        n_required_map: HashMap<NodeId, usize>,
        sharded_params_map: HashMap<NodeId, ShardedParams>,
    ) {
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

        // println!("nodes_sync_params : {:?}", self.nodes_sync_parameters);
        // println!("nodes_sync_params : {:?}", self.param_required_map);
    }

    fn on_register(&mut self, id: NodeId, tensor: CommunicationTensor<B>) {
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

                if *self
                    .param_required_map
                    .get(param_id)
                    .expect("Sharded tensor should have been registered by its device.")
                    == all_reduce_ops_queue.len()
                {
                    // println!("execute : {:?}", param_id);
                    B::all_reduce_inplace(
                        all_reduce_ops_queue.to_vec(),
                        burn_backend::AllReduceStrategy::Centralized,
                        burn_backend::ReduceOperation::Mean,
                    );
                    self.all_reduce_ops_queue.remove(param_id).unwrap();
                    self.param_required_map.remove(param_id).unwrap();
                }
            }
        }
    }

    fn is_finished(&mut self, sender: Sender<bool>) {
        if self.param_required_map.is_empty() {
            if !self.all_reduce_ops_queue.is_empty() {
                log::warn!("All reduce operations left hanging.")
            }
            sender.send(true).expect("Is finished channel open.")
        } else {
            sender.send(false).expect("Is finished channel open.")
        }
    }
}

// TODO: Tests
