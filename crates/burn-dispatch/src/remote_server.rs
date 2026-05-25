//! Concrete-backend dispatch for `burn-remote`'s server entry points.
//!
//! Lives in `burn-dispatch` because matching on [`DispatchDevice`] requires the
//! local `wgpu_metal`/`wgpu_vulkan`/`wgpu_webgpu` cfgs set by this crate's
//! `build.rs`, plus visibility of every in-tree `BackendIr` type. The user
//! surface (`Channel` enum, opaque `Device` argument) lives in `burn-tensor`.

use crate::DispatchDevice;
use crate::backends::*;

/// Start a websocket remote server, blocking the current thread.
///
/// The dispatch device determines which backend executes operations server-side.
/// Autodiff is stripped (the autodiff graph is a client-side concern); backends
/// that don't implement `BackendIr` (`LibTorch`, `Remote`) panic.
pub fn start_websocket(device: DispatchDevice, port: u16) {
    match device.inner() {
        #[cfg(feature = "cpu")]
        DispatchDevice::Cpu(device) => burn_remote::server::start_websocket::<Cpu>(device, port),
        #[cfg(feature = "cuda")]
        DispatchDevice::Cuda(device) => burn_remote::server::start_websocket::<Cuda>(device, port),
        #[cfg(feature = "rocm")]
        DispatchDevice::Rocm(device) => burn_remote::server::start_websocket::<Rocm>(device, port),
        #[cfg(feature = "wgpu")]
        DispatchDevice::Wgpu(device) => burn_remote::server::start_websocket::<Wgpu>(device, port),
        #[cfg(feature = "flex")]
        DispatchDevice::Flex(device) => burn_remote::server::start_websocket::<Flex>(device, port),
        #[cfg(any(feature = "ndarray", default_backend))]
        DispatchDevice::NdArray(device) => {
            burn_remote::server::start_websocket::<NdArray>(device, port)
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

/// Start a websocket remote server on the caller's tokio runtime.
///
/// See [`start_websocket`] for variant-handling rules.
pub async fn start_websocket_async(device: DispatchDevice, port: u16) {
    match device.inner() {
        #[cfg(feature = "cpu")]
        DispatchDevice::Cpu(device) => {
            burn_remote::server::start_websocket_async::<Cpu>(device, port).await
        }
        #[cfg(feature = "cuda")]
        DispatchDevice::Cuda(device) => {
            burn_remote::server::start_websocket_async::<Cuda>(device, port).await
        }
        #[cfg(feature = "rocm")]
        DispatchDevice::Rocm(device) => {
            burn_remote::server::start_websocket_async::<Rocm>(device, port).await
        }
        #[cfg(feature = "wgpu")]
        DispatchDevice::Wgpu(device) => {
            burn_remote::server::start_websocket_async::<Wgpu>(device, port).await
        }
        #[cfg(feature = "flex")]
        DispatchDevice::Flex(device) => {
            burn_remote::server::start_websocket_async::<Flex>(device, port).await
        }
        #[cfg(any(feature = "ndarray", default_backend))]
        DispatchDevice::NdArray(device) => {
            burn_remote::server::start_websocket_async::<NdArray>(device, port).await
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
