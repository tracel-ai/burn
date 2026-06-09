//! Concrete-backend dispatch for `burn-remote`'s server entry points.
//!
//! Lives in `burn-dispatch` because matching on [`DispatchDevice`] requires the
//! local `wgpu_metal`/`wgpu_vulkan`/`wgpu_webgpu` cfgs set by this crate's
//! `build.rs`, plus visibility of every in-tree `BackendIr` type. The user
//! surface (`Channel` enum, opaque `Device` argument) lives in `burn-tensor`.

use crate::backends::*;
use crate::{Dispatch, DispatchDevice, DispatchDeviceId};

/// Collect every [`Device<B>`] the host exposes for the backend that owns `$variant`, by
/// enumerating the backend (see [`Dispatch::enumerate`]) and unwrapping the matching variant.
/// `$id` is the [`DispatchDeviceId`] to enumerate; the result is `Vec<Device<B>>`, indexed by
/// hardware device index, which is exactly the index a client selects with `Device::remote`.
macro_rules! host_devices {
    ($id:expr, $variant:ident) => {
        Dispatch::enumerate($id)
            .into_iter()
            .filter_map(|device| match device {
                DispatchDevice::$variant(device) => Some(device),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
}

/// Start a websocket remote server, blocking the current thread.
///
/// The dispatch device selects which backend executes operations server-side; the server
/// then hosts **all** of that backend's devices (single host, multi-device), indexed by
/// hardware device index. Autodiff is stripped (the autodiff graph is a client-side
/// concern); backends that don't implement `BackendIr` (`LibTorch`, `Remote`) panic.
pub fn start_websocket(device: DispatchDevice, port: u16) {
    match device.inner() {
        #[cfg(feature = "cpu")]
        DispatchDevice::Cpu(_) => burn_remote::server::start_websocket::<Cpu>(
            host_devices!(DispatchDeviceId::Cpu, Cpu),
            port,
        ),
        #[cfg(feature = "cuda")]
        DispatchDevice::Cuda(_) => burn_remote::server::start_websocket::<Cuda>(
            host_devices!(DispatchDeviceId::Cuda, Cuda),
            port,
        ),
        #[cfg(feature = "metal")]
        DispatchDevice::Metal(_) => burn_remote::server::start_websocket::<Metal>(
            host_devices!(DispatchDeviceId::Metal, Metal),
            port,
        ),
        #[cfg(feature = "rocm")]
        DispatchDevice::Rocm(_) => burn_remote::server::start_websocket::<Rocm>(
            host_devices!(DispatchDeviceId::Rocm, Rocm),
            port,
        ),
        #[cfg(feature = "vulkan")]
        DispatchDevice::Vulkan(_) => {
            burn_remote::server::start_websocket::<Vulkan>(vec![WgpuDevice::DefaultDevice], port)
        }
        #[cfg(feature = "wgpu")]
        DispatchDevice::Wgpu(_) => burn_remote::server::start_websocket::<Wgpu>(
            host_devices!(DispatchDeviceId::Wgpu, Wgpu),
            port,
        ),
        #[cfg(feature = "webgpu")]
        DispatchDevice::WebGpu(_) => burn_remote::server::start_websocket::<WebGpu>(
            host_devices!(DispatchDeviceId::WebGpu, WebGpu),
            port,
        ),
        #[cfg(feature = "flex")]
        DispatchDevice::Flex(_) => burn_remote::server::start_websocket::<Flex>(
            host_devices!(DispatchDeviceId::Flex, Flex),
            port,
        ),
        #[cfg(any(feature = "ndarray", default_backend))]
        DispatchDevice::NdArray(_) => burn_remote::server::start_websocket::<NdArray>(
            host_devices!(DispatchDeviceId::NdArray, NdArray),
            port,
        ),
        #[cfg(feature = "tch")]
        DispatchDevice::LibTorch(_) => {
            panic!("LibTorch is not supported as a remote-server backend (no BackendIr impl)")
        }
        #[cfg(feature = "remote")]
        DispatchDevice::Remote(_) => {
            panic!("Cannot host a remote server on a remote device")
        }
        #[cfg(feature = "autodiff")]
        DispatchDevice::Autodiff(_) => {
            unreachable!("Autodiff stripped by .inner() above")
        }
    }
}

/// Start a websocket remote server on the caller's tokio runtime.
///
/// See [`start_websocket`] for variant-handling rules.
pub async fn start_websocket_async(device: DispatchDevice, port: u16) {
    match device.inner() {
        #[cfg(feature = "cpu")]
        DispatchDevice::Cpu(_) => {
            burn_remote::server::start_websocket_async::<Cpu>(
                host_devices!(DispatchDeviceId::Cpu, Cpu),
                port,
            )
            .await
        }
        #[cfg(feature = "cuda")]
        DispatchDevice::Cuda(_) => {
            burn_remote::server::start_websocket_async::<Cuda>(
                host_devices!(DispatchDeviceId::Cuda, Cuda),
                port,
            )
            .await
        }
        #[cfg(feature = "metal")]
        DispatchDevice::Metal(_) => {
            burn_remote::server::start_websocket_async::<Metal>(
                host_devices!(DispatchDeviceId::Metal, Metal),
                port,
            )
            .await
        }
        #[cfg(feature = "rocm")]
        DispatchDevice::Rocm(_) => {
            burn_remote::server::start_websocket_async::<Rocm>(
                host_devices!(DispatchDeviceId::Rocm, Rocm),
                port,
            )
            .await
        }
        #[cfg(feature = "vulkan")]
        DispatchDevice::Vulkan(_) => {
            burn_remote::server::start_websocket_async::<Vulkan>(
                vec![WgpuDevice::DefaultDevice],
                port,
            )
            .await
        }
        #[cfg(feature = "wgpu")]
        DispatchDevice::Wgpu(_) => {
            burn_remote::server::start_websocket_async::<Wgpu>(
                host_devices!(DispatchDeviceId::Wgpu, Wgpu),
                port,
            )
            .await
        }
        #[cfg(feature = "webgpu")]
        DispatchDevice::WebGpu(_) => {
            burn_remote::server::start_websocket_async::<WebGpu>(
                host_devices!(DispatchDeviceId::WebGpu, WebGpu),
                port,
            )
            .await
        }
        #[cfg(feature = "flex")]
        DispatchDevice::Flex(_) => {
            burn_remote::server::start_websocket_async::<Flex>(
                host_devices!(DispatchDeviceId::Flex, Flex),
                port,
            )
            .await
        }
        #[cfg(any(feature = "ndarray", default_backend))]
        DispatchDevice::NdArray(_) => {
            burn_remote::server::start_websocket_async::<NdArray>(
                host_devices!(DispatchDeviceId::NdArray, NdArray),
                port,
            )
            .await
        }
        #[cfg(feature = "tch")]
        DispatchDevice::LibTorch(_) => {
            panic!("LibTorch is not supported as a remote-server backend (no BackendIr impl)")
        }
        #[cfg(feature = "remote")]
        DispatchDevice::Remote(_) => {
            panic!("Cannot host a remote server on a remote device")
        }
        #[cfg(feature = "autodiff")]
        DispatchDevice::Autodiff(_) => {
            unreachable!("Autodiff stripped by .inner() above")
        }
    }
}
