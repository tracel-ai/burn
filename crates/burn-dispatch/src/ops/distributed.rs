use alloc::vec::Vec;
use core::mem::discriminant;

use crate::backends::*;

use burn_backend::{
    distributed::{
        DistributedBackend, DistributedConfig, DistributedParams, ReduceOperation, TensorRef,
    },
    tensor::FloatTensor,
};

use crate::{Dispatch, DispatchDevice};

macro_rules! dispatch_devices_arms {
    (
        $device:expr,
        $devices:expr,
        |$inner_devices:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {
        match $device {
            // Autodiff arm first
            #[cfg(feature = "autodiff")]
            $crate::DispatchDevice::Autodiff(inner) => {
                // Recursively dispatch on inner
                dispatch_devices_arms!(
                    @autodiff
                    &**inner,
                    $devices,
                    |$inner_devices| $body;
                    $([$Backend, $cfg]),*
                )
            },
            $(
                #[cfg($cfg)]
                $crate::DispatchDevice::$Backend(_) => {
                    assert!(
                        $devices
                            .iter()
                            .all(|d| discriminant(d) == discriminant($device)),
                        "All devices are expected to be of the same variant."
                    );
                    type B = $Backend<f32>;
                    let $inner_devices = $devices
                        .iter()
                        .map(|d| {
                            let DispatchDevice::$Backend(dev) = d else {
                                unreachable!()
                            };
                            dev.clone()
                        })
                        .collect();
                    $body
                }
            )*
        }
    };
    (
        @autodiff
        $device:expr,
        $devices:expr,
        |$inner_devices:ident| $body:expr;
        $([$Backend:ident, $cfg:meta]),*
    ) => {
        match $device {
            $(
                #[cfg($cfg)]
                $crate::DispatchDevice::$Backend(_) => {
                    assert!(
                        $devices
                            .iter()
                            .all(|d| discriminant(d) == discriminant($device)),
                        "All devices are expected to be of the same variant."
                    );
                    type B = Autodiff<$Backend<f32>>;
                    let $inner_devices = $devices
                        .iter()
                        .map(|d| {
                            let DispatchDevice::$Backend(dev) = d else {
                                unreachable!()
                            };
                            dev.clone()
                        })
                        .collect();
                    $body
                }
            )*
            $crate::DispatchDevice::Autodiff(_) => panic!("Autodiff should not wrap an autodiff device.")
        }
    };
}

/// Dispatches an operation body based on the provided devices.
macro_rules! dispatch_devices {
    ($device:expr, $devices:expr, |$inner_devices:ident| $body:expr) => {
        backend_list!(dispatch_devices_arms, $device, $devices, |$inner_devices| {
            $body
        })
    };
}

impl DistributedBackend for Dispatch {
    fn start_communication_server(devices: &[DispatchDevice], config: DistributedConfig) {
        if !devices.is_empty() {
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

    fn submit_sync_collective(device: &DispatchDevice) {
        dispatch_device!(device, |device| B::submit_sync_collective(device))
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
