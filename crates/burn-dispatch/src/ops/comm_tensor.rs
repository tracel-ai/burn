use std::mem::discriminant;

use crate::backends::*;

use burn_backend::{
    DistributedConfig, DistributedParams, ReduceOperation,
    ops::{CommunicationTensorOps, TensorRef},
    tensor::FloatTensor,
};

use crate::{Dispatch, DispatchDevice};

impl CommunicationTensorOps<Self> for Dispatch {
    fn start_communication_server(devices: Vec<DispatchDevice>, config: DistributedConfig) {
        if devices.len() > 0 {
            let first = &devices[0];
            dispatch_devices!(first, devices, |inner_devices| {
                B::start_communication_server(inner_devices, config)
            });
        }
    }

    fn close_communication_server(device: &DispatchDevice) {
        dispatch_device!(device, |device| B::close_communication_server(device))
    }

    fn register_sync_parameters(
        device: &DispatchDevice,
        sharded_param_ids: Vec<DistributedParams>,
    ) {
        dispatch_device!(device, |device| B::register_sync_parameters(
            device,
            sharded_param_ids,
        ))
    }

    fn submit_sync_collective(_device: &DispatchDevice) {
        unimplemented!()
    }

    fn submit_gradient_sync(_tensor: TensorRef<Self>, _distributed_params: DistributedParams) {
        unimplemented!()
    }

    unsafe fn all_reduce_in_place(_tensors: Vec<TensorRef<Self>>, _op: ReduceOperation) {
        unimplemented!()
    }

    fn sync_collective(_device: &DispatchDevice) {
        unimplemented!()
    }

    unsafe fn comm_device(_tensor: &TensorRef<Self>) -> DispatchDevice {
        unimplemented!()
    }

    unsafe fn float_from_ref(_tensor: &TensorRef<Self>) -> FloatTensor<Self> {
        unimplemented!()
    }
}
