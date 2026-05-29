use alloc::vec::Vec;
use core::mem::discriminant;

use burn_backend::{
    DeviceId,
    distributed::{
        CollectiveTensor, DistributedBackend, DistributedConfig, DistributedParams,
        ReduceOperation, TensorRef,
    },
    tensor::FloatTensor,
};

// TODO: REMOVE.
use std::any::type_name;

fn print_type_of<T>(_: &T) {
    println!("{}", type_name::<T>());
}

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
                // Recursively dispatch on inner
                let inner_devices = $devices
                    .iter()
                    .map(|d| {
                        // Dynamically match against all possible backends passed to the macro
                        let inn = match &d {
                            #[cfg(feature = "autodiff")]
                            $crate::DispatchDevice::Autodiff(d_inner) => d_inner.clone(),
                            _ => unreachable!("All devices are expected to be of the same variant."),
                        };
                        println!("inn: {:?}", inn);
                        println!("inner: {:?}", inner);
                        inn
                    })
                    .collect::<Vec<_>>();
                let inner_devices = inner_devices.as_slice();
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
                    // print_type_of($device);
                    // print_type_of(&$device.clone());
                    // $devices
                    //     .iter().for_each(|d| {
                    //         print_type_of(d);
                    //         println!("{:?}", d);
                    //         println!("{:?}", $device);
                    //         println!("{:?}", discriminant(d));
                    //         println!("{:?}", discriminant($device));
                    // });
                    assert!(
                        $devices
                            .iter()
                            .all(|d| discriminant(d) == discriminant($device)),
                        "All devices are expected to be of the same variant."
                    );
                    type B = $crate::backends::$Backend;
                    // let $inner_devices = $devices
                    //     .iter()
                    //     .map(|d| {
                    //         let DispatchDevice::$Backend(dev) = d else {
                    //             unreachable!()
                    //         };
                    //         dev.clone()
                    //     })
                    //     .collect::<Vec<_>>();
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
        // macro_rules! __inner_match {
        //     ($val:expr) => {
        //         match $val {
        //             $(
        //                 #[$cfg]
        //                 $crate::DispatchDevice::$Backend(device_ident) => device_ident.clone(),
        //             )*
        //             _ => unreachable!("Invalid backend device mapping"),
        //         }
        //     };
        // }
        match $device {
            $(
                #[cfg($cfg)]
                $crate::DispatchDevice::$Backend(inner) => {
                    print_type_of($device);
                    print_type_of(&$device.clone());
                    $devices
                        .iter().for_each(|d| {
                            print_type_of(d);
                            println!("{:?}", d);
                            println!("{:?}", $device);
                            // println!("{:?}", discriminant(d));
                            println!("{:?}", discriminant($device));
                    });
                    // assert!(
                    //     $devices
                    //         .iter()
                    //         .all(|d| discriminant(d) == discriminant($device)),
                    //     "All devices are expected to be of the same variant."
                    // );
                    type B = $crate::backends::Autodiff<$crate::backends::$Backend>;
                    // let $inner_devices = $devices.clone();
                    let $inner_devices = $devices
                        .iter()
                        .map(|d| {
                            let inn = d.inner.clone();
                            println!("{:?}", inn);
                            inn

                            // match &d.inner.clone() {
                            //     $(
                            //         #[cfg($cfg)]
                            //         $crate::DispatchDevice::$Backend(device_ident) => device_ident.clone(),
                            //     )*
                            //     // Handle the fallback if it's nested autodiff or unsupported
                            //     _ => unreachable!("Invalid backend device mapping"),
                            // }
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

macro_rules! __inner_match {
    ($val:expr, $([$Backend:ident, $cfg:meta]),*) => {
        match $val {
            $(
                #[$cfg]
                $crate::DispatchDevice::$Backend(device_ident) => device_ident.clone(),
            )*
            _ => unreachable!("Invalid backend device mapping"),
        }
    };
}

/// Dispatches an operation body based on the provided devices.
macro_rules! dispatch_distributed_devices {
    (@distributed $device:expr, $devices:expr, |$inner_devices:ident| $body:expr) => {
        dispatch_distributed_devices!(@internal distributed_backend_list,
            $device,
            $devices,
            |$inner_devices| $body
        )
    };
    (@internal $list_macro:ident, $device:expr, $devices:expr, |$inner_devices:ident| $body:expr) => {
        $list_macro!(dispatch_distributed_devices_arms, $device, $devices, |$inner_devices| $body)
    };
}

impl DistributedBackend for Dispatch {
    fn start_communication_server(devices: &[Self::Device], config: DistributedConfig) {
        if !devices.is_empty() {
            let first = &devices[0];
            dispatch_distributed_devices!(@distributed first, devices, |inner_devices| {
                // B::start_communication_server(&inner_devices, config)
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
        let tensor = unary_float!(@distributed tensor, float, |tensor| {
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
