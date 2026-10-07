//! Serving Burn's own backends: the backend is the one the devices belong to.
//!
//! Lives in `burn-dispatch` because matching on [`DispatchDevice`] requires the local
//! `cube_backend` cfg set by this crate's `build.rs`, plus visibility of every in-tree `BackendIr`
//! type. The user surface is `burn::server::RemoteServer`.

#[cfg(not(target_family = "wasm"))]
use burn_remote::server::Transport;
use burn_remote::{
    Endpoint,
    server::{BackendServer, RemoteProtocol, ServeError, ServerSettings},
};

use crate::DispatchDevice;
use crate::backends::*;

/// Bind `$b` to the one backend every device of `$devices` belongs to, and `$hosted` to the
/// devices as that backend's own, then run `$body`. Autodiff is stripped: the autodiff graph is the
/// client's.
macro_rules! with_backend {
    ($devices:expr, |$b:ident, $hosted:ident| $body:expr) => {{
        let devices: Vec<DispatchDevice> =
            $devices.into_iter().map(DispatchDevice::inner).collect();
        match devices.first() {
            None => Err(ServeError::NoDevices),
            #[cfg(cube_backend)]
            Some(DispatchDevice::Cube(_)) => {
                type $b = Cube;
                let $hosted = hosted!(devices, Cube)?;
                $body
            }
            #[cfg(feature = "flex")]
            Some(DispatchDevice::Flex(_)) => {
                type $b = Flex;
                let $hosted = hosted!(devices, Flex)?;
                $body
            }
            #[cfg(feature = "tch")]
            Some(DispatchDevice::LibTorch(_)) => Err(ServeError::UnsupportedDevice {
                reason: "LibTorch cannot run remote sessions".into(),
            }),
            #[cfg(feature = "remote")]
            Some(DispatchDevice::Remote(_)) => Err(ServeError::UnsupportedDevice {
                reason: "a remote device cannot host a remote server".into(),
            }),
            #[cfg(feature = "capture")]
            Some(DispatchDevice::Capture(_)) => Err(ServeError::UnsupportedDevice {
                reason: "a capture device cannot host a remote server".into(),
            }),
            #[cfg(feature = "autodiff")]
            Some(DispatchDevice::Autodiff(_)) => {
                unreachable!("Autodiff stripped by DispatchDevice::inner")
            }
        }
    }};
}

/// `$devices` as `$variant`'s own devices, or [`ServeError::MixedBackends`].
macro_rules! hosted {
    ($devices:expr, $variant:ident) => {
        $devices
            .into_iter()
            .map(|device| match device {
                DispatchDevice::$variant(device) => Ok(device),
                #[allow(unreachable_patterns)]
                _ => Err(ServeError::MixedBackends),
            })
            .collect::<Result<Vec<_>, ServeError>>()
    };
}

/// Serve `devices` on `transport`, blocking until Ctrl+C or `SIGTERM`.
#[cfg(not(target_family = "wasm"))]
pub fn serve(
    devices: Vec<DispatchDevice>,
    settings: ServerSettings,
    transport: Transport,
) -> Result<(), ServeError> {
    with_backend!(devices, |B, hosted| BackendServer::<B>::new(hosted)
        .with_settings(settings)
        .serve(transport))
}

/// Serve `devices` on `transport` until the returned future is dropped.
#[cfg(not(target_family = "wasm"))]
pub async fn serve_async(
    devices: Vec<DispatchDevice>,
    settings: ServerSettings,
    transport: Transport,
) -> Result<(), ServeError> {
    with_backend!(devices, |B, hosted| BackendServer::<B>::new(hosted)
        .with_settings(settings)
        .serve_async(transport)
        .await)
}

/// Burn Remote's handler for an application's Iroh router on `endpoint`, serving `devices`.
pub fn into_protocol(
    devices: Vec<DispatchDevice>,
    settings: ServerSettings,
    endpoint: &Endpoint,
) -> Result<RemoteProtocol, ServeError> {
    with_backend!(devices, |B, hosted| BackendServer::<B>::new(hosted)
        .with_settings(settings)
        .into_protocol(endpoint))
}
