//! Remote-execution server entry points.
//!
//! Hosts a Burn server that executes tensor operations on behalf of remote clients. The backend is
//! selected by the Device passed in; the transport is selected by Channel. Iroh is the primary
//! transport; WebSocket is retained for compatibility.
//!
//! Two serving modes:
//!
//! - Turnkey (start / start_async): no Iroh exposure. Pick an identity with RemoteSecret and pass
//!   it in a Channel::Iroh; clients dial its public id.
//! - Composed (protocol): for applications that own their own Iroh router. Burn hands back only
//!   its protocol handler to register alongside the application's own protocols.
//!
//! User-defined backends that implement BackendIr but are not part of DispatchDevice use
//! burn_remote::server::RemoteServerBuilder directly; that is also how custom operations
//! (backend extensions) are hosted, over either transport.

use std::sync::Arc;

use crate::Device;
pub use burn_dispatch::backends::remote::server::{
    AllowAll, AuthorizationRequest, PeerAuthorizer, RemoteProtocol,
};
pub use burn_dispatch::backends::remote::telemetry;
pub use burn_dispatch::backends::remote::{Endpoint, RemoteSecret};
pub use burn_dispatch::devices::BURN_REMOTE_ALPN;

use telemetry::TelemetryProbe;

/// Transport used to serve remote clients. Re-exported from `burn-remote` (via `burn-dispatch`) so
/// the whole stack shares one definition.
pub use burn_dispatch::remote_server::Channel;

/// Build Burn's protocol handler for `device`'s backend.
///
/// Returns a builder to optionally attach telemetry and an authorizer before calling `build`.
/// Register the result on an application-owned Iroh router under BURN_REMOTE_ALPN.
pub fn protocol(device: Device, endpoint: &Endpoint) -> RemoteProtocolBuilder<'_> {
    RemoteProtocolBuilder::new(device, endpoint)
}

/// Configures optional telemetry and authorization for a RemoteProtocol handler.
pub struct RemoteProtocolBuilder<'a> {
    device: Device,
    endpoint: &'a Endpoint,
    probe: Option<TelemetryProbe>,
    authorizer: Option<Arc<dyn PeerAuthorizer>>,
}

impl<'a> RemoteProtocolBuilder<'a> {
    /// Create a new builder for `device`'s backend, to register on `endpoint`.
    pub fn new(device: Device, endpoint: &'a Endpoint) -> Self {
        Self {
            device,
            endpoint,
            probe: None,
            authorizer: None,
        }
    }

    /// Attach a telemetry probe for per-session monitoring. Pair with
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
    pub fn build(self) -> RemoteProtocol {
        burn_dispatch::remote_server::remote_protocol(
            self.device.into_dispatch(),
            self.endpoint,
            self.probe.unwrap_or_else(TelemetryProbe::disabled),
            self.authorizer.unwrap_or_else(|| Arc::new(AllowAll)),
        )
    }
}

impl<'a> From<RemoteProtocolBuilder<'a>> for RemoteProtocol {
    fn from(builder: RemoteProtocolBuilder<'a>) -> Self {
        builder.build()
    }
}

/// Start a remote-execution server, blocking the current thread.
///
/// The backend is determined by `device`: e.g. `Device::cuda(0)` runs ops on
/// CUDA, `Device::flex()` on the Flex CPU backend. Autodiff devices are
/// transparently stripped; the autodiff graph is a client-side concern.
///
/// # Panics
///
/// Panics if `device` selects a backend that doesn't support remote execution
/// (currently `LibTorch`, or a `Remote` device; hosting on a remote device
/// makes no sense).
#[cfg(not(target_family = "wasm"))]
pub fn start(device: Device, channel: Channel) {
    burn_dispatch::remote_server::start(device.into_dispatch(), channel)
}

/// Start a remote-execution server on the caller's async runtime.
///
/// See [`start`] for backend-selection rules.
#[cfg(not(target_family = "wasm"))]
pub async fn start_async(device: Device, channel: Channel) {
    burn_dispatch::remote_server::start_async(device.into_dispatch(), channel).await
}
