//! Concrete-backend dispatch for `burn-remote`'s server entry points.
//!
//! Lives in `burn-dispatch` because matching on [`DispatchDevice`] requires the
//! local `cube_backend`/`default_backend` cfgs set by this crate's `build.rs`,
//! plus visibility of every in-tree `BackendIr` type. The user surface
//! (`Channel` enum, opaque `Device` argument) lives in `burn-tensor`.

use std::sync::Arc;

use burn_remote::Endpoint;
use burn_remote::server::{CustomOpRegistry, IrohRemoteProtocol, PeerAuthorizer, RemoteProtocol};
use burn_remote::telemetry::TelemetryProbe;

use crate::DispatchDevice;
use crate::backends::*;
// Only the non-cubecl backends still enumerate through `Dispatch`; the cubecl one enumerates
// its devices directly (see `host_devices!`).
#[cfg(any(feature = "flex", feature = "ndarray", default_backend))]
use crate::{Dispatch, DispatchDeviceId};

/// Transport used to serve remote clients. Re-exported from `burn-remote` so the whole stack
/// shares one definition.
pub use burn_remote::server::Channel;

/// Collect every [`Device<B>`] the host exposes for the backend that owns `$variant`, by
/// enumerating the backend (see [`Dispatch::enumerate`]) and unwrapping the matching variant.
/// `$id` is the [`DispatchDeviceId`] to enumerate; the result is `Vec<Device<B>>`, indexed by
/// hardware device index, which is exactly the index a client selects with `Device::remote`.
///
/// Enumeration is only trustworthy when it finds **more than one** device: several backends (the
/// wgpu family and the CPU-only ones like Flex) can't enumerate hardware and report either
/// nothing or a single placeholder index that isn't the device you'd actually run on. In that
/// case fall back to hosting a single device rather than a possibly-empty or bogus enumerated
/// list.
macro_rules! host_devices {
    // The cubecl runtimes are one backend, so `Dispatch::enumerate` lists every device of every
    // runtime the build compiled in. Hosting a CUDA server should not hand clients the wgpu and
    // CPU devices found alongside it, so narrow the list to the runtime `$device` names, and fall
    // back to `$device` itself — the runtime-specific default the caller already chose — rather
    // than to `Device::default()`, which is whichever runtime cubecl ranks highest.
    (cube: $device:expr) => {{
        let devices = $crate::backend::cube_devices($device.runtime());
        if devices.len() > 1 {
            devices
        } else {
            vec![$device.clone()]
        }
    }};
    ($id:expr, $variant:ident) => {{
        let devices = Dispatch::enumerate($id)
            .into_iter()
            .filter_map(|device| match device {
                DispatchDevice::$variant(device) => Some(device),
                #[allow(unreachable_patterns)]
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
/// share this one match instead of duplicating a `#[cfg]`-gated arm per backend each. Backends
/// without a `BackendIr` impl (`LibTorch`, `Remote`) panic; `Autodiff` is already stripped by
/// `.inner()`.
macro_rules! with_backend {
    ($device:expr, |$b:ident, $devices:ident| $body:expr) => {
        match $device.inner() {
            #[cfg(cube_backend)]
            DispatchDevice::Cube(device) => {
                type $b = Cube;
                let $devices = host_devices!(cube: device);
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
            #[cfg(feature = "capture")]
            DispatchDevice::Capture(_) => {
                panic!("Cannot host a remote server on a capture device")
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
/// `device` selects the backend; `channel` selects the transport. The server hosts that backend's
/// devices, indexed by hardware device index. Use [`start_async`] for the async counterpart.
#[cfg(not(target_family = "wasm"))]
pub fn start(device: DispatchDevice, channel: Channel) {
    with_backend!(device, |B, devices| {
        burn_remote::server::RemoteServerBuilder::<B>::new(devices)
            .channel(channel)
            .start()
    })
}

/// Async counterpart of [`start`]; runs on the caller's tokio runtime instead of blocking.
#[cfg(not(target_family = "wasm"))]
pub async fn start_async(device: DispatchDevice, channel: Channel) {
    with_backend!(device, |B, devices| {
        burn_remote::server::RemoteServerBuilder::<B>::new(devices)
            .channel(channel)
            .start_async()
            .await
    })
}

/// Build a backend-erased protocol handler for `device`'s backend.
///
/// Resolves the dispatch device to its concrete backend type and returns a [`RemoteProtocol`]
/// the caller registers on its own Iroh router under [`BURN_REMOTE_ALPN`].
pub fn remote_protocol(
    device: DispatchDevice,
    endpoint: &Endpoint,
    probe: TelemetryProbe,
    authorizer: Arc<dyn PeerAuthorizer>,
) -> RemoteProtocol {
    with_backend!(device, |B, devices| {
        RemoteProtocol::new(IrohRemoteProtocol::<B>::new(
            endpoint.clone(),
            devices,
            authorizer,
            probe,
            CustomOpRegistry::default(),
        ))
    })
}
