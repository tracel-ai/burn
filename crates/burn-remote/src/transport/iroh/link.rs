//! Iroh implementations of the session-link frame traits.
//!
//! An Iroh session is one bidirectional QUIC stream; its two halves are already separate owned
//! values (`SendStream` / `RecvStream`), so they map directly onto [`FrameSink`] / [`FrameSource`].
//! A frame is its length as a little-endian `u64`, then its bytes.

use bytes::Bytes;
use iroh::endpoint::{ReadExactError, RecvStream, SendStream};

use crate::{
    shared::BUFFERS,
    transport::link::{FrameSink, FrameSource, MAX_WHOLE_MESSAGE_SIZE},
};

impl FrameSink for SendStream {
    async fn send(&mut self, frame: Bytes) -> Result<(), String> {
        let length = Bytes::copy_from_slice(&(frame.len() as u64).to_le_bytes());
        // Chunks are handed to QUIC as they are; `write_all` would copy the frame into its buffer.
        self.write_all_chunks(&mut [length, frame])
            .await
            .map_err(|err| format!("Failed to write Iroh frame: {err}"))
    }

    async fn close(&mut self) -> Result<(), String> {
        self.finish()
            .map_err(|err| format!("Failed to finish Iroh stream: {err}"))
    }
}

impl FrameSource for RecvStream {
    async fn recv(&mut self, max_len: usize) -> Result<Option<Bytes>, String> {
        let Some(length) = FrameLength::read(self).await? else {
            return Ok(None);
        };
        let length = length.at_most(max_len)?;
        if length <= MAX_WHOLE_MESSAGE_SIZE {
            let mut frame = vec![0; length];
            self.read_exact(&mut frame)
                .await
                .map_err(|err| format!("Failed to read Iroh frame: {err}"))?;
            return Ok(Some(frame.into()));
        }
        let mut frame = BUFFERS.take_to_overwrite(length).map_err(|_| {
            format!("Peer sent a frame of {length} bytes, more than can be allocated")
        })?;
        frame.resize(length, 0);
        self.read_exact(&mut frame)
            .await
            .map_err(|err| format!("Failed to read Iroh frame: {err}"))?;
        Ok(Some(frame.into_bytes()))
    }

    async fn recv_into(&mut self, buf: &mut [u8]) -> Result<usize, String> {
        let length = FrameLength::read(self)
            .await?
            .ok_or_else(|| "Peer closed the stream in the middle of a message".to_string())?
            .at_most(buf.len())?;
        self.read_exact(&mut buf[..length])
            .await
            .map_err(|err| format!("Failed to read Iroh frame: {err}"))?;
        Ok(length)
    }
}

/// The length a frame opens with, as the peer claims it.
struct FrameLength(u64);

impl FrameLength {
    /// The next frame's length, or `None` when the peer finished the stream between frames.
    async fn read(stream: &mut RecvStream) -> Result<Option<Self>, String> {
        let mut length = [0u8; 8];
        match stream.read_exact(&mut length).await {
            Ok(()) => Ok(Some(Self(u64::from_le_bytes(length)))),
            Err(ReadExactError::FinishedEarly(0)) => Ok(None),
            Err(err) => Err(format!("Failed to read Iroh frame length: {err}")),
        }
    }

    /// Refused past `max_len` before any of the frame is read.
    fn at_most(self, max_len: usize) -> Result<usize, String> {
        usize::try_from(self.0)
            .ok()
            .filter(|length| *length <= max_len)
            .ok_or_else(|| {
                format!(
                    "Peer sent an oversized Burn Remote frame: {} bytes (max {max_len})",
                    self.0
                )
            })
    }
}
