//! Remote-execution server entry points.
//!
//! Hosts a Burn server that executes tensor operations on behalf of remote clients. The backend is
//! selected by the [`Device`] passed in (the same handle tensor ops use); the transport is selected
//! by [`Channel`]. Iroh is the primary transport; WebSocket is retained for compatibility.
//!
//! Two ways to serve, for the two kinds of user:
//!
//! - **Turnkey** ([`start`] / [`start_async`]): no exposure to Iroh at all. Pick an identity with
//!   [`RemoteSecret`] (the counterpart to choosing a port for WebSocket); clients dial its
//!   [`id`](RemoteSecret::id).
//! - **Composed** ([`protocol`]): for an application that owns its own Iroh [`Router`]. Burn hands
//!   back only its protocol handler, to register alongside the application's own protocols.
//!
//! ```rust,ignore
//! use burn::{Device, server::{start, Channel, RemoteSecret}};
//!
//! let secret = RemoteSecret::random();
//! println!("clients dial: {}", secret.id());
//! start(Device::default(), Channel::Iroh { secret });
//! ```
//!
//! User-defined backends that implement `BackendIr` but aren't part of `DispatchDevice` build a
//! `burn_remote::server::RemoteServerBuilder` directly with the concrete backend type; that is
//! also how custom operations (backend extensions) are hosted, over either transport.

use std::sync::Arc;

use crate::Device;
pub use burn_dispatch::backends::remote::server::{
    AuthorizationRequest, PeerAuthorizer, RemoteProtocol, Router, RouterBuilder,
};
pub use burn_dispatch::backends::remote::telemetry;
pub use burn_dispatch::backends::remote::{Endpoint, RemoteSecret};
pub use burn_dispatch::devices::BURN_REMOTE_ALPN;

use telemetry::TelemetryProbe;

/// Transport used to serve remote clients. Re-exported from `burn-remote` (via `burn-dispatch`) so
/// the whole stack shares one definition.
pub use burn_dispatch::remote_server::Channel;

/// Burn's protocol handler for `device`'s backend, to register on an application-owned Iroh router.
///
/// This is the composition entry: Burn exposes only its protocol, and the application builds and
/// owns the [`Router`], registering Burn under [`BURN_REMOTE_ALPN`] alongside its own protocols.
/// Optionally attach telemetry or an authorization policy before building the handler.
///
/// ```rust,ignore
/// let handler = burn::server::protocol(device, &endpoint).with_telemetry(probe).handler();
/// let router = Router::builder(endpoint)
///     .accept(BURN_REMOTE_ALPN, handler)
///     .accept(MY_ALPN, my_protocol)
///     .spawn();
/// ```
pub fn protocol(device: Device, endpoint: &Endpoint) -> RemoteProtocolBuilder<'_> {
    RemoteProtocolBuilder {
        device,
        endpoint,
        probe: None,
        authorizer: None,
    }
}

/// Configures the optional telemetry and authorization policy for [`protocol`], then builds the
/// backend-erased [`RemoteProtocol`] handler.
pub struct RemoteProtocolBuilder<'a> {
    device: Device,
    endpoint: &'a Endpoint,
    probe: Option<TelemetryProbe>,
    authorizer: Option<Arc<dyn PeerAuthorizer>>,
}

impl<'a> RemoteProtocolBuilder<'a> {
    /// Emit per-session telemetry into `probe` for live monitoring. Pair with
    /// [`telemetry::TelemetryProbe::channel`] to obtain a subscription a dashboard can drain.
    pub fn with_telemetry(mut self, probe: TelemetryProbe) -> Self {
        self.probe = Some(probe);
        self
    }

    /// Authorize or reject each incoming compute session. The policy receives the peer identity,
    /// the requested device index, and the opaque credential carried by the client's ticket.
    pub fn with_authorizer(mut self, authorizer: impl PeerAuthorizer) -> Self {
        self.authorizer = Some(Arc::new(authorizer));
        self
    }

    /// Build the backend-erased protocol handler.
    pub fn handler(self) -> RemoteProtocol {
        burn_dispatch::remote_server::remote_protocol(
            self.device.into_dispatch(),
            self.endpoint,
            self.probe,
            self.authorizer,
        )
    }
}

impl<'a> From<RemoteProtocolBuilder<'a>> for RemoteProtocol {
    fn from(builder: RemoteProtocolBuilder<'a>) -> Self {
        builder.handler()
    }
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
#[cfg(not(target_family = "wasm"))]
pub fn start(device: Device, channel: Channel) {
    match channel {
        Channel::Iroh { secret } => {
            burn_dispatch::remote_server::start_iroh(device.into_dispatch(), secret)
        }
        #[cfg(feature = "remote-websocket")]
        Channel::WebSocket { port } => {
            burn_dispatch::remote_server::start_websocket(device.into_dispatch(), port)
        }
    }
}

/// Start a remote-execution server on the caller's async runtime.
///
/// See [`start`] for backend-selection rules.
#[cfg(not(target_family = "wasm"))]
pub async fn start_async(device: Device, channel: Channel) {
    match channel {
        Channel::Iroh { secret } => {
            burn_dispatch::remote_server::start_iroh_async(device.into_dispatch(), secret).await
        }
        #[cfg(feature = "remote-websocket")]
        Channel::WebSocket { port } => {
            burn_dispatch::remote_server::start_websocket_async(device.into_dispatch(), port).await
        }
    }
}
