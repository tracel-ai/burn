use std::collections::HashMap;

use crate::{Backend, TensorMetadata};
use crate::{DeviceId, DeviceOps, tensor::Device};

use crate::distributed::{
    DistributedConfig, DistributedParamId, DistributedParams, TensorRef,
    client::DistributedSyncMessage,
};

pub(crate) struct DistributedSyncServer<B: Backend> {
    config: DistributedConfig,
    all_reduce_ops_queue: HashMap<DistributedParamId, Vec<TensorRef<B>>>,
    param_required_map: HashMap<DistributedParamId, usize>,
    num_devices: usize,
    devices_registered: usize,
    syncing_devices: Vec<Device<B>>,
    callbacks: HashMap<DeviceId, oneshot::Sender<()>>,
}

impl<B: Backend> DistributedSyncServer<B> {
    /// Create a new gradient sync server instance.
    pub(crate) fn new(num_devices: usize, config: DistributedConfig) -> Self {
        Self {
            config,
            all_reduce_ops_queue: HashMap::default(),
            param_required_map: HashMap::default(),
            num_devices,
            devices_registered: 0,
            syncing_devices: vec![],
            callbacks: HashMap::default(),
        }
    }

    /// Process message from client.
    pub(crate) fn process_message(&mut self, msg: DistributedSyncMessage<B>) {
        match msg {
            DistributedSyncMessage::RegisterSyncParameters(params) => {
                self.register_sync_params(params)
            }
            DistributedSyncMessage::TensorSync((tensor, params)) => {
                self.register_tensor(tensor, params)
            }
            DistributedSyncMessage::CollectiveSync((device, callback)) => {
                self.collective_sync(device, callback)
            }
        }
    }

    /// Called at the start of the backward process. Lets the device announce what parameters are nodes in the autodiff graph and how many times they are required.
    fn register_sync_params(&mut self, sharded_params: Vec<DistributedParams>) {
        sharded_params.iter().for_each(|params| {
            *self.param_required_map.entry(params.param_id).or_insert(0) += 1;
        });
        self.devices_registered += 1;
    }

    /// Called on registration of a gradient. Calls the all_reduce operation for any parameter that is no longer required in the autodiff graph.
    fn register_tensor(&mut self, tensor: TensorRef<B>, sharded_params: DistributedParams) {
        let op_queue = self
            .all_reduce_ops_queue
            .entry(sharded_params.param_id)
            .or_insert(vec![]);
        op_queue.push(tensor.clone());
        self.launch_ops();
    }

    fn collective_sync(&mut self, device: Device<B>, callback: oneshot::Sender<()>) {
        self.callbacks.insert(device.id(), callback);
        self.syncing_devices.push(device.clone());
        self.try_launch_sync();
    }

    fn try_launch_sync(&mut self) {
        // All at once: a device released early registers its next backward before this reset.
        if self.syncing_devices.len() < self.num_devices || !self.all_reduce_ops_queue.is_empty() {
            return;
        }
        self.devices_registered = 0;
        self.param_required_map.clear();
        for device in self.syncing_devices.drain(..) {
            let callback = self.callbacks.remove(&device.id()).unwrap();
            callback.send(()).expect("Can send callback");
        }
    }

    fn launch_ops(&mut self) {
        if self.devices_registered == self.num_devices {
            for (param_id, num_tensors) in self.param_required_map.clone() {
                let queued_tensors = self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);

                if num_tensors == queued_tensors.len() {
                    // Safety: Tensors sent to the `DistributedSyncServer` should not be accessed or modified until the end of the backward pass.
                    // A native group takes its members in ascending device order, and every member
                    // receives the same result in any.
                    queued_tensors.sort_by_key(|tensor| unsafe { &*tensor.0 }.device().id());
                    let tensors = queued_tensors
                        .iter()
                        .map(|tensor| unsafe { (*tensor.0).clone() })
                        .collect();
                    let reduced = B::float_all_reduce(tensors, self.config.all_reduce_op);

                    // Safety: each device waits in `submit_sync_collective` for every reduction
                    // before reading its gradients.
                    unsafe {
                        queued_tensors
                            .iter()
                            .zip(reduced)
                            .for_each(|(tensor_ref, reduced)| *tensor_ref.0 = reduced);
                    }

                    self.all_reduce_ops_queue.remove(&param_id).unwrap();
                    self.param_required_map.remove(&param_id).unwrap();
                    self.try_launch_sync();
                }
            }
        }
    }
}
