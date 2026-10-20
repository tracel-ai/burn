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
    devices: Vec<DeviceId>,
    devices_registered: usize,
    /// The backwards waiting to be released, by device.
    callbacks: HashMap<DeviceId, Vec<oneshot::Sender<()>>>,
}

impl<B: Backend> DistributedSyncServer<B> {
    /// Create a new gradient sync server instance.
    pub(crate) fn new(devices: &[Device<B>], config: DistributedConfig) -> Self {
        Self {
            config,
            all_reduce_ops_queue: HashMap::default(),
            param_required_map: HashMap::default(),
            devices: devices.iter().map(DeviceOps::id).collect(),
            devices_registered: 0,
            callbacks: HashMap::default(),
        }
    }

    /// Process message from client.
    pub(crate) fn process_message(&mut self, msg: DistributedSyncMessage<B>) {
        match msg {
            DistributedSyncMessage::RegisterSyncParameters((device, params)) => {
                self.register_sync_params(device, params)
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
    fn register_sync_params(&mut self, device: Device<B>, sharded_params: Vec<DistributedParams>) {
        if !self.is_member(&device) {
            return;
        }
        sharded_params.iter().for_each(|params| {
            *self.param_required_map.entry(params.param_id).or_insert(0) += 1;
        });
        self.devices_registered += 1;
    }

    /// Called on registration of a gradient. Calls the all_reduce operation for any parameter that is no longer required in the autodiff graph.
    fn register_tensor(&mut self, tensor: TensorRef<B>, sharded_params: DistributedParams) {
        // Safety: its backward sends `CollectiveSync` after this and waits for the answer, so the
        // gradient is alive.
        if !self.is_member(&unsafe { &*tensor.0 }.device()) {
            return;
        }
        let op_queue = self
            .all_reduce_ops_queue
            .entry(sharded_params.param_id)
            .or_insert(vec![]);
        op_queue.push(tensor.clone());
        self.launch_ops();
    }

    fn collective_sync(&mut self, device: Device<B>, callback: oneshot::Sender<()>) {
        if !self.is_member(&device) {
            callback.send(()).expect("Can send callback");
            return;
        }
        self.callbacks
            .entry(device.id())
            .or_default()
            .push(callback);
        self.try_launch_sync();
    }

    /// Whether `device` is in the group. Another device's backward syncs nothing here: none of
    /// its gradients are queued, so it is released at once.
    fn is_member(&self, device: &Device<B>) -> bool {
        self.devices.contains(&device.id())
    }

    fn try_launch_sync(&mut self) {
        // All at once: a device released early registers its next backward before this reset.
        let devices_missing = self.callbacks.len() < self.devices.len();
        if devices_missing || !self.all_reduce_ops_queue.is_empty() {
            return;
        }
        self.devices_registered = 0;
        self.param_required_map.clear();
        for callback in self.callbacks.drain().flat_map(|(_, callbacks)| callbacks) {
            callback.send(()).expect("Can send callback");
        }
    }

    fn launch_ops(&mut self) {
        if self.devices_registered == self.devices.len() {
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
