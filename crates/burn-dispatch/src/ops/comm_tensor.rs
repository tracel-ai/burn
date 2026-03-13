use std::mem::discriminant;

use crate::backends::*;

use burn_backend::{
    ShardedParams,
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

    fn init_collective_queue(device: &DispatchDevice, sharded_param_ids: Vec<ShardedParams>) {
        dispatch_device!(device, |device| B::init_collective_queue(
            device,
            sharded_param_ids,
        ))
    }

    fn collective_sync(device: &DispatchDevice) {
        dispatch_device!(device, |device| B::collective_sync(device,))
    }

    // TODO:
    fn all_reduce_in_place(_tensor: TensorRef<Self>, _sharded_params: ShardedParams) {
        todo!()
    }

    fn supports_native_collective(device: &DispatchDevice) -> bool {
        dispatch_device!(device, |device| B::supports_native_collective(device,))
    }
}
