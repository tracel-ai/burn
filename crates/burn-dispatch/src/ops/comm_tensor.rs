use std::mem::discriminant;

use crate::backends::*;

use burn_backend::{
    DistributedParams,
    ops::{CommunicationTensorOps, TensorRef},
};

use crate::{Dispatch, DispatchDevice};

impl CommunicationTensorOps<Self> for Dispatch {
    fn start_communication_server(devices: Vec<DispatchDevice>) {
        if devices.len() > 0 {
            let first = &devices[0];
            dispatch_devices!(first, devices, |inner_devices| {
                B::start_communication_server(inner_devices)
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

    fn sync_collective(device: &DispatchDevice) {
        dispatch_device!(device, |device| B::sync_collective(device,))
    }

    // TODO:
    fn submit_gradient_sync(_tensor: TensorRef<Self>, _distributed_params: DistributedParams) {
        todo!()
    }

    fn supports_native_collective(device: &DispatchDevice) -> bool {
        dispatch_device!(device, |device| B::supports_native_collective(device,))
    }
}
