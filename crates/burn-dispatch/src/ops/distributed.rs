use alloc::vec::Vec;

use burn_backend::{
    DeviceId,
    distributed::{
        CollectiveTensor, DistributedConfig, DistributedOps, DistributedParams, ReduceOperation,
        TensorRef,
    },
    tensor::FloatTensor,
};

use crate::{Dispatch, DispatchDevice};

macro_rules! dispatch_distributed_devices_arms {
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
                let inner_devices = $devices
                    .iter()
                    .map(|d| {
                         match &d {
                            #[cfg(feature = "autodiff")]
                            $crate::DispatchDevice::Autodiff(d_inner) => *d_inner.inner.clone(),
                            _ => unreachable!("All devices are expected to be of the same variant."),
                        }
                    })
                    .collect::<Vec<_>>();
                let inner_devices = inner_devices.as_slice();
                // Recursively dispatch on inner
                dispatch_distributed_devices_arms!(
                    @autodiff
                    &**inner,
                    &*inner_devices,
                    |$inner_devices| $body;
                    $([$Backend, $cfg]),*
                )
            },
            $(
                #[cfg($cfg)]
                $crate::DispatchDevice::$Backend(_) => {
                    type B = $crate::backends::$Backend;
                    let $inner_devices = $devices
                        .iter()
                        .map(|d| {
                            let DispatchDevice::$Backend(dev) = d else {
                                unreachable!("All devices are expected to be of the same variant.")
                            };
                            dev.clone()
                        })
                        .collect::<Vec<_>>();
                    $body
                }
            )*
            other => panic!("Distributed operations are not supported for device {other:?}"),
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
                    type B = $crate::backends::Autodiff<$crate::backends::$Backend>;
                    let $inner_devices = $devices
                        .iter()
                        .map(|d| {
                            let DispatchDevice::$Backend(dev) = d else {
                                unreachable!("All devices are expected to be of the same variant.")
                            };
                            dev.clone()
                        })
                        .collect::<Vec<_>>();
                    $body
                }
            )*
            $crate::DispatchDevice::Autodiff(_) => panic!("Autodiff should not wrap an autodiff device."),
            other => panic!("Distributed operations are not supported for device {other:?}"),
        }
    };
}

/// Dispatches an operation body based on the provided devices.
macro_rules! dispatch_distributed_devices {
    ($device:expr, $devices:expr, |$inner_devices:ident| $body:expr) => {
        distributed_backend_list!(
            dispatch_distributed_devices_arms,
            $device,
            $devices,
            |$inner_devices| $body
        )
    };
}

// In builds without a collective-capable backend (Cuda/Remote), the distributed dispatch arms
// are all cfg'd out, leaving only a diverging fallback — so the captured arguments and trailing
// expressions are intentionally unused/unreachable.
#[allow(unused_variables, unreachable_code)]
impl DistributedOps<Self> for Dispatch {
    fn start_communication_server(devices: &[DispatchDevice], config: DistributedConfig) {
        if !devices.is_empty() {
            let first = &devices[0];
            dispatch_distributed_devices!(first, devices, |inner_devices| {
                B::start_communication_server(&inner_devices, config)
            });
        }
    }

    fn close_communication_server(device: &DispatchDevice) {
        dispatch_device!(@distributed device, |device| {
            B::close_communication_server(device)
        })
    }

    fn register_sync_parameters(
        device: &DispatchDevice,
        sharded_param_ids: Vec<DistributedParams>,
    ) {
        dispatch_device!(@distributed device, |device| B::register_sync_parameters(
            device,
            sharded_param_ids,
        ))
    }

    fn submit_sync_collective(device: &DispatchDevice) {
        dispatch_device!(@distributed device, |device| B::submit_sync_collective(device))
    }

    fn submit_gradient_sync(_tensor: TensorRef<Self>, _distributed_params: DistributedParams) {
        unimplemented!()
    }

    fn all_reduce(
        tensor: FloatTensor<Self>,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> CollectiveTensor<Self> {
        // Safety: we call `assume_resolved` only to wrap it in a new `CollectiveTensor`.
        // Explicit type: the distributed dispatch only emits arms for collective-capable
        // backends (Cuda, Remote), so a build with none of them leaves only the diverging
        // fallback and the match would otherwise infer `!`.
        let tensor: FloatTensor<Self> = unary_float!(@distributed tensor, float, |tensor| {
            let collective_tensor = B::all_reduce(tensor, op, device_ids);
            unsafe { collective_tensor.assume_resolved() }
        } => Float);
        CollectiveTensor::new(tensor)
    }

    fn sync_collective(device: &DispatchDevice) {
        dispatch_device!(@distributed device, |device| B::sync_collective(device))
    }

    unsafe fn comm_device(_tensor: &TensorRef<Self>) -> DispatchDevice {
        unimplemented!()
    }

    unsafe fn float_from_ref(_tensor: &TensorRef<Self>) -> FloatTensor<Self> {
        unimplemented!()
    }
}
