//! The session-link abstraction.
//!
//! A session is a duplex, length-framed message channel: the client submits a stream of
//! [`RemoteMessage`](crate::shared::RemoteMessage)s and the server returns a stream of
//! [`TaskResponse`](crate::shared::TaskResponse)s. Every transport realizes this as one
//! bidirectional stream, split into an outgoing [`FrameSink`] and an incoming [`FrameSource`] so
//! the response-writer task can own the sink while the request-reader loop owns the source.
//!
//! Frames are opaque `Bytes` here; encoding/decoding to the protocol types lives in the session
//! pump (server) and client service, so the transport layer only moves bytes.

use bytes::Bytes;
use core::future::Future;

/// `Send` on native targets, unconstrained in the browser.
///
/// Iroh streams are `!Send` on wasm (they live on the JS event loop), so the link traits cannot
/// require `Send` unconditionally. The real `Send` requirement is applied where session tasks are
/// spawned, via the cfg'd [`spawn_detached`](crate::server::spawn::spawn_detached) /
/// [`Executor`](crate::client::service::Executor) helpers — exactly as the concrete channel enums
/// did before this abstraction existed.
#[cfg(not(target_family = "wasm"))]
pub(crate) trait MaybeSend: Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send + ?Sized> MaybeSend for T {}
#[cfg(target_family = "wasm")]
pub(crate) trait MaybeSend {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSend for T {}

/// The outgoing half of a session link: writes length-framed messages to the peer.
pub(crate) trait FrameSink: MaybeSend + 'static {
    /// Send one already-encoded frame.
    fn send(&mut self, frame: Bytes) -> impl Future<Output = Result<(), String>> + MaybeSend;

    /// Finish the stream; no more frames will be sent.
    fn close(&mut self) -> impl Future<Output = Result<(), String>> + MaybeSend;
}

/// The incoming half of a session link: reads length-framed messages from the peer.
pub(crate) trait FrameSource: MaybeSend + 'static {
    /// Receive the next frame, or `None` when the peer closes the stream cleanly.
    fn recv(&mut self) -> impl Future<Output = Result<Option<Bytes>, String>> + MaybeSend;
}
