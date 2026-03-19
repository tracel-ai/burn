use cubecl::device::DeviceId;
use std::collections::HashMap;

use crate::DeviceOps;
use crate::client::GradientSyncMessage;
use crate::ops::TensorRef;
use crate::tensor::Device;
use crate::{Backend, DistributedParams, ModuleParamId, ReduceOperation};

pub(crate) struct GradientSyncServer<B: Backend> {
    all_reduce_ops_queue: HashMap<ModuleParamId, Vec<TensorRef<B>>>,
    param_required_map: HashMap<ModuleParamId, usize>,
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
            GradientSyncMessage::RegisterDevice(params) => self.register_device(params),
            GradientSyncMessage::Register((tensor, params)) => self.on_register(tensor, params),
            GradientSyncMessage::Sync((device, callback)) => self.sync(device, callback),
        }
    }

    fn try_flush_sync(&mut self) {
        if self.all_reduce_ops_queue.is_empty() && !self.syncing_devices.is_empty() {
            for d in self.syncing_devices.clone() {
                let callback = self.callbacks.remove(&d.id()).unwrap();
                let closure = Box::new(move || B::collective_sync_native(&d));
                callback.send(closure).expect("Can send callback");
                self.devices_synced += 1;
            }
            self.syncing_devices.clear();
        }

        if self.devices_synced == self.num_devices {
            self.devices_registered = 0;
            self.syncing_devices.clear();
            self.devices_synced = 0;
            self.all_reduce_ops_queue.clear();
            self.param_required_map.clear();
            self.callbacks.clear();
        }
    }

    fn sync(&mut self, device: Device<B>, callback: oneshot::Sender<Box<dyn FnOnce() + Send>>) {
        self.callbacks.insert(device.id(), callback);
        self.syncing_devices.push(device.clone());
        self.try_flush_sync();
    }

    /// Called at the start of the backward process. Lets the device announce what parameters are nodes in the autodiff graph and how many times they are required.
    fn register_device(&mut self, sharded_params: Vec<DistributedParams>) {
        sharded_params.iter().for_each(|param| {
            let id = param
                .param_id
                .expect("Sharded parameters should have a module parameter ID.");
            *self.param_required_map.entry(id).or_insert(0) += 1;
        });
        self.devices_registered += 1;
    }

    fn launch_ops(&mut self) {
        if self.devices_registered == self.num_devices {
            for (param_id, num_tensors) in self.param_required_map.clone() {
                let all_reduce_ops_queue =
                    self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);

                if num_tensors == all_reduce_ops_queue.len() {
                    B::all_reduce_in_place_native(
                        all_reduce_ops_queue.clone(),
                        ReduceOperation::Mean, // TODO: hard coded
                    );
                    self.all_reduce_ops_queue.remove(&param_id).unwrap();
                    self.param_required_map.remove(&param_id).unwrap();
                    self.try_flush_sync();
                }
            }
        }
    }

    /// Called on registration of a gradient. Calls the all_reduce operation for any parameter that is no longer required in the autodiff graph.
    fn on_register(&mut self, tensor: TensorRef<B>, sharded_params: DistributedParams) {
        let param_id = sharded_params
            .param_id
            .expect("Sharded tensor should have a parameter ID.");
        let all_reduce_ops_queue = self.all_reduce_ops_queue.entry(param_id).or_insert(vec![]);
        all_reduce_ops_queue.push(tensor.clone());
        self.launch_ops();
    }
}

// TODO: Tests
