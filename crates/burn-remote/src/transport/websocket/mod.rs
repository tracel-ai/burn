//! WebSocket transport implementation — the legacy address-and-port transport.
//!
//! Self-contained: the session uses one full-duplex socket (split into [`FrameSink`]/[`FrameSource`]
//! halves in [`link`]) driven by the shared session pump, and the turnkey server lives in [`server`].
//!
//! [`FrameSink`]: crate::transport::link::FrameSink
//! [`FrameSource`]: crate::transport::link::FrameSource

mod link;

#[cfg(feature = "server")]
mod transfer;

#[cfg(not(target_family = "wasm"))]
mod server;
#[cfg(not(target_family = "wasm"))]
pub(crate) use server::start_websocket_async;
