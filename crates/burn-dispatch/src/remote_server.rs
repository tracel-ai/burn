//! Concrete-backend dispatch for `burn-remote`'s server entry points.
//!
//! Lives in `burn-dispatch` because matching on [`DispatchDevice`] requires the
//! local `wgpu_metal`/`wgpu_vulkan`/`wgpu_webgpu` cfgs set by this crate's
//! `build.rs`, plus visibility of every in-tree `BackendIr` type. The user
//! surface (`Channel` enum, opaque `Device` argument) lives in `burn-tensor`.

use std::sync::Arc;

use burn_remote::server::{PeerAuthorizer, RemoteProtocol};
use burn_remote::telemetry::TelemetryProbe;
use burn_remote::{Endpoint, RemoteNode};

use crate::backends::*;
use crate::{Dispatch, DispatchDevice, DispatchDeviceId};

/// Transport used to serve remote clients. Re-exported from `burn-remote` so the whole stack
/// shares one definition.
pub use burn_remote::server::Channel;

/// Collect every [`Device<B>`] the host exposes for the backend that owns `$variant`, by
/// enumerating the backend (see [`Dispatch::enumerate`]) and unwrapping the matching variant.
/// `$id` is the [`DispatchDeviceId`] to enumerate; the result is `Vec<Device<B>>`, indexed by
/// hardware device index, which is exactly the index a client selects with `Device::remote`.
///
/// Enumeration is only trustworthy when it finds **more than one** device: several backends (the
/// wgpu family — Vulkan/WebGpu/Wgpu — and the CPU-only ones like Flex) can't enumerate hardware
/// and report either nothing or a single placeholder index that isn't the device you'd actually
/// run on. In that case fall back to hosting a single backend-specific default device
/// (`Device::<B>::default()`, e.g. `WgpuDevice::DefaultDevice`) rather than a possibly-empty or
/// bogus enumerated list. This generalizes what used to be a hardcoded Vulkan special-case.
macro_rules! host_devices {
    ($id:expr, $variant:ident) => {{
        let devices = Dispatch::enumerate($id)
            .into_iter()
            .filter_map(|device| match device {
                DispatchDevice::$variant(device) => Some(device),
                _ => None,
            })
            .collect::<Vec<_>>();
        if devices.len() > 1 {
            devices
        } else {
            vec![Default::default()]
        }
    }};
}

/// Run `$body` with the concrete backend that owns `$device`'s variant bound to the type alias
/// `$b` and that backend's host device list bound to `$devices`.
///
/// This is the single source of truth for the `DispatchDevice` → concrete-`BackendIr` mapping.
/// The sync and async server entry points differ only in whether `$body` awaits the call, so they
/// share this one match instead of duplicating eleven `#[cfg]`-gated arms each. Backends without a
/// `BackendIr` impl (`LibTorch`, `Remote`) panic; `Autodiff` is already stripped by `.inner()`.
macro_rules! with_backend {
    ($device:expr, |$b:ident, $devices:ident| $body:expr) => {
        match $device.inner() {
            #[cfg(feature = "cpu")]
            DispatchDevice::Cpu(_) => {
                type $b = Cpu;
                let $devices = host_devices!(DispatchDeviceId::Cpu, Cpu);
                $body
            }
            #[cfg(feature = "cuda")]
            DispatchDevice::Cuda(_) => {
                type $b = Cuda;
                let $devices = host_devices!(DispatchDeviceId::Cuda, Cuda);
                $body
            }
            #[cfg(feature = "metal")]
            DispatchDevice::Metal(_) => {
                type $b = Metal;
                let $devices = host_devices!(DispatchDeviceId::Metal, Metal);
                $body
            }
            #[cfg(feature = "rocm")]
            DispatchDevice::Rocm(_) => {
                type $b = Rocm;
                let $devices = host_devices!(DispatchDeviceId::Rocm, Rocm);
                $body
            }
            #[cfg(feature = "vulkan")]
            DispatchDevice::Vulkan(_) => {
                type $b = Vulkan;
                let $devices = host_devices!(DispatchDeviceId::Vulkan, Vulkan);
                $body
            }
            #[cfg(feature = "wgpu")]
            DispatchDevice::Wgpu(_) => {
                type $b = Wgpu;
                let $devices = host_devices!(DispatchDeviceId::Wgpu, Wgpu);
                $body
            }
            #[cfg(feature = "webgpu")]
            DispatchDevice::WebGpu(_) => {
                type $b = WebGpu;
                let $devices = host_devices!(DispatchDeviceId::WebGpu, WebGpu);
                $body
            }
            #[cfg(any(feature = "flex", default_backend))]
            DispatchDevice::Flex(_) => {
                type $b = Flex;
                let $devices = host_devices!(DispatchDeviceId::Flex, Flex);
                $body
            }
            #[cfg(feature = "ndarray")]
            DispatchDevice::NdArray(_) => {
                type $b = NdArray;
                let $devices = host_devices!(DispatchDeviceId::NdArray, NdArray);
                $body
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
    };
}

/// Start a remote-execution server for `device`'s backend, blocking the current thread.
///
/// The dispatch device selects which backend executes operations server-side; `channel` selects the
/// transport. The server hosts that backend's devices (single host, multi-device), indexed by
/// hardware device index. See [`with_backend`] for how the backend is resolved, [`host_devices`]
/// for how its device list is chosen, and [`start_async`] for the async counterpart.
///
/// Both transports run through the one [`RemoteServerBuilder`](burn_remote::server::RemoteServerBuilder)
/// entry point; the builder picks the protocol from `channel`.
#[cfg(not(target_family = "wasm"))]
pub fn start(device: DispatchDevice, channel: Channel) {
    with_backend!(device, |B, devices| {
        burn_remote::server::RemoteServerBuilder::<B>::new(devices)
            .channel(channel)
            .start()
    })
}

/// Start a remote-execution server for `device`'s backend on the caller's tokio runtime.
///
/// The async counterpart of [`start`]; the two share the same backend-resolution match (see
/// [`with_backend`]) and differ only in awaiting the server future.
#[cfg(not(target_family = "wasm"))]
pub async fn start_async(device: DispatchDevice, channel: Channel) {
    with_backend!(device, |B, devices| {
        burn_remote::server::RemoteServerBuilder::<B>::new(devices)
            .channel(channel)
            .start_async()
            .await
    })
}

/// Build a backend-erased Burn Remote protocol handler for `device`'s backend.
///
/// Resolves the dispatch device to its concrete `BackendIr` (see [`with_backend`]) and erases it
/// behind [`RemoteProtocol`], so the `burn-tensor` user surface never names a backend. `probe` and
/// `authorizer` are applied when present. The caller registers the result on an Iroh router (alone,
/// or alongside other protocols) under [`BURN_REMOTE_ALPN`](burn_remote::BURN_REMOTE_ALPN).
pub fn remote_protocol(
    device: DispatchDevice,
    endpoint: &Endpoint,
    probe: Option<TelemetryProbe>,
    authorizer: Option<Arc<dyn PeerAuthorizer>>,
) -> RemoteProtocol {
    let node = RemoteNode::from_endpoint(endpoint.clone());
    with_backend!(device, |B, devices| {
        let mut protocol = node.protocol::<B>(devices);
        if let Some(probe) = probe {
            protocol = protocol.with_telemetry(probe);
        }
        if let Some(authorizer) = authorizer {
            protocol = protocol.with_authorizer_arc(authorizer);
        }
        RemoteProtocol::new(protocol)
    })
}
