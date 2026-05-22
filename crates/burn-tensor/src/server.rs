//! Remote-execution server entry points.
//!
//! Hosts a Burn server that executes tensor operations on behalf of remote
//! clients. The backend is selected by the [`Device`] passed in (the same
//! device handle used by tensor ops); the transport is selected by [`Channel`].
//!
//! ```rust,ignore
//! use burn::{Device, server::{start, Channel}};
//!
//! fn main() {
//!     start(Device::default(), Channel::WebSocket { port: 3000 });
//! }
//! ```
//!
//! User-defined backends that implement `BackendIr` but aren't part of
//! `DispatchDevice` should call burn_remote::server::start_websocket.
//! directly with the concrete backend type parameter.

use crate::Device;

/// Transport used to serve remote clients.
#[derive(Debug, Clone, Copy)]
pub enum Channel {
    /// WebSocket server bound to `0.0.0.0:port`.
    WebSocket {
        /// Port to bind on.
        port: u16,
    },
}

/// Start a remote-execution server, blocking the current thread.
///
/// The backend is determined by `device`: e.g. `Device::cuda(0)` runs ops on
/// CUDA, `Device::flex()` on the Flex CPU backend. Autodiff devices are
/// transparently stripped — the autodiff graph is a client-side concern.
///
/// # Panics
///
/// Panics if `device` selects a backend that doesn't support remote execution
/// (currently `LibTorch`, or a `Remote` device — hosting on a remote device
/// makes no sense).
pub fn start(device: Device, channel: Channel) {
    match channel {
        Channel::WebSocket { port } => {
            burn_dispatch::remote_server::start_websocket(device.into_dispatch(), port)
        }
    }
}

/// Start a remote-execution server on the caller's async runtime.
///
/// See [`start`] for backend-selection rules.
pub async fn start_async(device: Device, channel: Channel) {
    match channel {
        Channel::WebSocket { port } => {
            burn_dispatch::remote_server::start_websocket_async(device.into_dispatch(), port).await
        }
    }
}
