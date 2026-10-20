//! The session-link abstraction.
//!
//! A session is a duplex link: the client submits a stream of
//! [`RemoteMessage`](crate::shared::RemoteMessage)s and the server returns a stream of
//! [`TaskResponse`](crate::shared::TaskResponse)s, each carried in frames. Every transport
//! realizes this as one bidirectional stream, split into an outgoing [`FrameSink`] and an incoming
//! [`FrameSource`] so the response-writer task can own the sink while the request-reader loop owns
//! the source.
//!
//! Frames are opaque `Bytes` here; encoding/decoding to the protocol types lives in the session
//! pump (server) and client service, so the transport layer only moves bytes. Past the handshake,
//! [`message`](super::message) carries each message in as many frames as it takes.

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

/// The largest frame either transport carries; a longer message travels in several.
pub const MAX_FRAME_SIZE: usize = 1024 * 1024;

/// The largest message sent whole, copied into one frame behind its tag. It is copied anyway, so
/// it is built and read in a plain allocation; a larger message travels in segments of their own
/// and is worth a pooled buffer.
pub const MAX_WHOLE_MESSAGE_SIZE: usize = 64 * 1024;

/// The largest frame read from a peer before it is authorized: a stream's header, a session's
/// `Init` or a tensor-transfer request. It bounds what a stranger can make an Iroh server hold;
/// WebSocket reads a frame up to [`MAX_FRAME_SIZE`] before refusing it.
pub const MAX_UNAUTHORIZED_FRAME_SIZE: usize = 64 * 1024;

/// The outgoing half of a session link: writes frames to the peer.
pub(crate) trait FrameSink: MaybeSend + 'static {
    /// Send one already-encoded frame.
    fn send(&mut self, frame: Bytes) -> impl Future<Output = Result<(), String>> + MaybeSend;

    /// Finish the stream; no more frames will be sent.
    fn close(&mut self) -> impl Future<Output = Result<(), String>> + MaybeSend;
}

/// The incoming half of a session link: reads frames from the peer.
pub(crate) trait FrameSource: MaybeSend + 'static {
    /// Receive the next frame, refused past `max_len` bytes, or `None` when the peer closes the
    /// stream cleanly.
    fn recv(
        &mut self,
        max_len: usize,
    ) -> impl Future<Output = Result<Option<Bytes>, String>> + MaybeSend;

    /// Receive the next frame into the start of `buf` and return its length, refusing one longer
    /// than `buf` or the stream ending first. A transport that hands each frame over in a buffer
    /// of its own copies it in once.
    fn recv_into(
        &mut self,
        buf: &mut [u8],
    ) -> impl Future<Output = Result<usize, String>> + MaybeSend {
        async move {
            let frame = self
                .recv(buf.len())
                .await?
                .ok_or_else(|| "Peer closed the stream in the middle of a message".to_string())?;
            buf[..frame.len()].copy_from_slice(&frame);
            Ok(frame.len())
        }
    }
}
