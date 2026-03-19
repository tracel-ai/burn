use cubecl::device::DeviceId;
use std::collections::HashMap;

use crate::DeviceOps;
use crate::client::GradientSyncMessage;
use crate::ops::TensorRef;
use crate::tensor::Device;
use crate::{Backend, DistributedParamId, DistributedParams, ReduceOperation};

pub(crate) struct GradientSyncServer<B: Backend> {
    all_reduce_ops_queue: HashMap<DistributedParamId, Vec<TensorRef<B>>>,
    param_required_map: HashMap<DistributedParamId, usize>,
    num_devices: usize,
    devices_registered: usize,
    syncing_devices: Vec<Device<B>>,
    devices_synced: usize,
    callbacks: HashMap<DeviceId, oneshot::Sender<Box<dyn FnOnce() + Send>>>,
}

impl<B: Backend> GradientSyncServer<B> {
    /// Create a new gradient sync server instance.
    pub(crate) fn new(devices: Vec<B::Device>) -> Self {
        Self {
            all_reduce_ops_queue: HashMap::default(),
            param_required_map: HashMap::default(),
            num_devices: devices.len(),
            devices_registered: 0,
            syncing_devices: vec![],
            devices_synced: 0,
            callbacks: HashMap::default(),
        }
    }

    /// Process message from client.
    pub(crate) fn process_message(&mut self, msg: GradientSyncMessage<B>) {
        match msg {
            GradientSyncMessage::RegisterSyncParameters(params) => {
                self.register_sync_params(params)
            }
            GradientSyncMessage::GradientSync((tensor, params)) => {
                self.register_gradient(tensor, params)
            }
            GradientSyncMessage::CollectiveSync((device, callback)) => {
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
    fn register_gradient(&mut self, tensor: TensorRef<B>, sharded_params: DistributedParams) {
        let op_queue = self
            .all_reduce_ops_queue
            .entry(sharded_params.param_id)
            .or_insert(vec![]);
        op_queue.push(tensor.clone());
        self.launch_ops();
    }

    fn collective_sync(
        &mut self,
        device: Device<B>,
        callback: oneshot::Sender<Box<dyn FnOnce() + Send>>,
    ) {
        self.callbacks.insert(device.id(), callback);
        self.syncing_devices.push(device.clone());
        self.try_launch_sync();
    }

    fn try_launch_sync(&mut self) {
        if self.all_reduce_ops_queue.is_empty() && !self.syncing_devices.is_empty() {
            for d in self.syncing_devices.clone() {
                let callback = self.callbacks.remove(&d.id()).unwrap();
                let closure = Box::new(move || B::sync_collective(&d));
                callback.send(closure).expect("Can send callback");
                self.devices_synced += 1;
            }
            self.syncing_devices.clear();
        }

        if self.devices_synced == self.num_devices {
            self.devices_registered = 0;
            self.devices_synced = 0;
            self.param_required_map.clear();
            self.callbacks.clear();
        }
    }

    fn launch_ops(&mut self) {
        if self.devices_registered == self.num_devices {
            for (param_id, num_tensors) in self.param_required_map.clone() {
                let queued_tensors = self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);

                if num_tensors == queued_tensors.len() {
                    B::all_reduce_in_place(
                        queued_tensors.clone(),
                        ReduceOperation::Mean, // TODO: hard coded
                    );
                    self.all_reduce_ops_queue.remove(&param_id).unwrap();
                    self.param_required_map.remove(&param_id).unwrap();
                    self.try_launch_sync();
                }
            }
        }
    }
}
