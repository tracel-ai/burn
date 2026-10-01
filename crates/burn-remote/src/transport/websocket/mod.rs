//! WebSocket transport implementation: the simplest setup on a trusted network.
//!
//! Self-contained: the session uses one full-duplex socket (split into [`FrameSink`]/[`FrameSource`]
//! halves in [`link`]) driven by the shared session pump, and the server lives in [`server`].
//!
//! [`FrameSink`]: crate::transport::link::FrameSink
//! [`FrameSource`]: crate::transport::link::FrameSource

mod link;

#[cfg(feature = "server")]
mod transfer;

#[cfg(not(target_family = "wasm"))]
mod server;
#[cfg(not(target_family = "wasm"))]
pub use server::WebSocketTransport;
