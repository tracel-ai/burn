use std::{collections::HashMap, mem::discriminant};

use crate::backends::*;

use burn_backend::{
    ModuleParamId, PeerId, ReduceOperation, ShardedParams,
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
            // assert!(
            //     devices
            //         .iter()
            //         .all(|d| discriminant(d) == discriminant(first)),
            //     "All devices are expected to be of the same variant."
            // );
            // // dispatch_device!(&first, |device| start_gradient_sync_server::<B>(devices))
            // match devices[0] {
            //     DispatchDevice::Cuda(_) => {
            //         let devs = devices
            //             .iter()
            //             .map(|d| {
            //                 let DispatchDevice::Cuda(dev) = d else {
            //                     unreachable!()
            //                 };
            //                 dev.clone()
            //             })
            //             .collect();
            //         type B = Cuda<f32>;
            //         B::start_communication_server(devs);
            //     }
            //     DispatchDevice::NdArray(_) => {
            //         let devs = devices
            //             .iter()
            //             .map(|d| {
            //                 let DispatchDevice::NdArray(dev) = d else {
            //                     unreachable!()
            //                 };
            //                 dev.clone()
            //             })
            //             .collect();
            //         type B = NdArray<f32>;
            //         B::start_communication_server(devs);
            //     }
            //     DispatchDevice::Autodiff(_) => {
            //         // let devs = devices
            //         //     .iter()
            //         //     .map(|d| {
            //         //         let DispatchDevice::Autodiff(dev) = d else {
            //         //             unreachable!()
            //         //         };
            //         //         dev.clone()
            //         //     })
            //         //     .collect();
            //         // type B = Autodiff<f32>;
            //         // B::start_communication_server(devs);
            //     }
            // };
        }
    }

    fn close_communication_server(device: &DispatchDevice) {
        dispatch_device!(device, |device| B::close_communication_server(device))
    }

    fn register_graph(
        device: &DispatchDevice,
        // n_required_map: HashMap<u64, usize>,
        // sharded_params_map: HashMap<u64, ShardedParams>,
        sharded_param_ids: Vec<ShardedParams>,
    ) {
        dispatch_device!(device, |device| B::register_graph(
            device,
            // n_required_map,
            // sharded_params_map
            sharded_param_ids,
        ))
    }

    fn communication_sync(device: &DispatchDevice) {
        dispatch_device!(device, |device| B::communication_sync(device,))
    }

    fn all_reduce_inplace(tensor: TensorRef<Self>, sharded_params: ShardedParams) {
        todo!()
        // unsafe {
        //     ref_op!(
        //         **tensor.0,
        //         B::all_reduce_inplace(tensor, peer_id, all_ids, op)
        //     )
        // }
    }

    /// If this backend supports native communication operations e.g. NCCL for Cuda.
    /// TODO: ARGS and returns
    fn supports_native_communication(device: &DispatchDevice) -> bool {
        dispatch_device!(device, |device| B::supports_native_communication(device,))
    }
}
