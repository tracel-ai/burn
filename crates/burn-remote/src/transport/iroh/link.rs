//! Iroh implementations of the session-link frame traits.
//!
//! An Iroh session is one bidirectional QUIC stream; its two halves are already separate owned
//! values (`SendStream` / `RecvStream`), so they map directly onto [`FrameSink`] / [`FrameSource`].

use bytes::Bytes;
use iroh::endpoint::{RecvStream, SendStream};

use super::node::{recv_frame, send_frame};
use crate::transport::link::{FrameSink, FrameSource};

impl FrameSink for SendStream {
    async fn send(&mut self, frame: Bytes) -> Result<(), String> {
        send_frame(self, &frame).await
    }

    async fn close(&mut self) -> Result<(), String> {
        self.finish()
            .map_err(|err| format!("Failed to finish Iroh stream: {err}"))
    }
}

impl FrameSource for RecvStream {
    async fn recv(&mut self) -> Result<Option<Bytes>, String> {
        recv_frame(self).await.map(|frame| frame.map(Bytes::from))
    }
}
