//! WebSocket implementations of the session-link frame traits.
//!
//! A WebSocket session is one full-duplex socket; `burn_communication` splits it into independent
//! send/receive halves, which map directly onto [`FrameSink`] / [`FrameSource`]. The inherent
//! `send`/`recv`/`close` on each half do the binary framing; here we only adapt the message type
//! (`Message` ↔ `Bytes`) and the error type (`String`), and hold a frame to the reader's limit.

use bytes::Bytes;

use burn_communication::Message;
use burn_communication::websocket::{WsClientSink, WsClientStream, WsServerSink, WsServerStream};

use crate::transport::link::{FrameSink, FrameSource};

impl FrameSink for WsServerSink {
    async fn send(&mut self, frame: Bytes) -> Result<(), String> {
        WsServerSink::send(self, Message::new(frame))
            .await
            .map_err(|err| err.to_string())
    }

    async fn close(&mut self) -> Result<(), String> {
        WsServerSink::close(self)
            .await
            .map_err(|err| err.to_string())
    }
}

impl FrameSource for WsServerStream {
    async fn recv(&mut self, max_len: usize) -> Result<Option<Bytes>, String> {
        WsServerStream::recv(self)
            .await
            .map_err(|err| err.to_string())?
            .map(|message| ReceivedFrame(message).at_most(max_len))
            .transpose()
    }
}

impl FrameSink for WsClientSink {
    async fn send(&mut self, frame: Bytes) -> Result<(), String> {
        WsClientSink::send(self, Message::new(frame))
            .await
            .map_err(|err| err.to_string())
    }

    async fn close(&mut self) -> Result<(), String> {
        WsClientSink::close(self)
            .await
            .map_err(|err| err.to_string())
    }
}

impl FrameSource for WsClientStream {
    async fn recv(&mut self, max_len: usize) -> Result<Option<Bytes>, String> {
        WsClientStream::recv(self)
            .await
            .map_err(|err| err.to_string())?
            .map(|message| ReceivedFrame(message).at_most(max_len))
            .transpose()
    }
}

/// A frame the socket has already read in full, held to a reader's limit only once it arrived.
struct ReceivedFrame(Message);

impl ReceivedFrame {
    fn at_most(self, max_len: usize) -> Result<Bytes, String> {
        let frame = self.0.data;
        if frame.len() > max_len {
            return Err(format!(
                "Peer sent an oversized Burn Remote frame: {} bytes (max {max_len})",
                frame.len()
            ));
        }
        Ok(frame)
    }
}
