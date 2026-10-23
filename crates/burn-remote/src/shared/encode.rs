use std::{
    collections::TryReserveError,
    io::{self, Write},
};

use bytes::Bytes;
use rmp_serde::encode::Error;
use serde::Serialize;

use super::buffer::{BUFFERS, PooledBuffer};
use crate::transport::link::{MAX_FRAME_SIZE, MAX_WHOLE_MESSAGE_SIZE};

/// A message's MessagePack encoding, as sent to a peer.
pub trait Encode: Serialize + Sized {
    /// The message's MessagePack bytes. Takes the message so its tensor data is freed before the
    /// bytes go out, not after.
    fn encode(self) -> Result<Encoded, Error> {
        let mut writer = SegmentWriter::default();
        rmp_serde::encode::write(&mut writer, &self)?;
        Ok(writer.finish())
    }
}

/// A message's bytes in segments of at most one frame, each its own buffer, so each is released
/// once it is sent.
#[derive(Clone, Debug)]
pub struct Encoded {
    segments: Vec<Bytes>,
    len: usize,
}

impl Encoded {
    pub fn len(&self) -> usize {
        self.len
    }

    /// The segments in order, none empty and none longer than [`MAX_FRAME_SIZE`].
    pub fn into_segments(self) -> impl Iterator<Item = Bytes> {
        self.segments.into_iter()
    }

    /// The bytes in one buffer, copied only when they span several segments.
    pub fn into_bytes(self) -> Bytes {
        match <[Bytes; 1]>::try_from(self.segments) {
            Ok([segment]) => segment,
            Err(segments) => segments.concat().into(),
        }
    }
}

/// Collects an encoding into segments of at most a frame. A message sent whole grows in a plain
/// `Vec`; a larger one is written into buffers from [`BUFFERS`], the first grown until it holds a
/// frame and every later one a frame filled to its capacity.
#[derive(Default)]
struct SegmentWriter {
    small: Vec<u8>,
    full: Vec<Bytes>,
    open: Option<PooledBuffer>,
}

impl SegmentWriter {
    fn finish(self) -> Encoded {
        let mut segments = self.full;
        if !self.small.is_empty() {
            segments.push(self.small.into());
        }
        segments.extend(self.open.map(PooledBuffer::into_bytes));
        let len = segments.iter().map(Bytes::len).sum();
        Encoded { segments, len }
    }

    /// The open segment, after growing or closing it if it is full and taking a buffer if none is
    /// open; the first buffer taken starts with what was written while the message was small.
    fn open_segment(&mut self, wanted: usize) -> Result<&mut PooledBuffer, TryReserveError> {
        if let Some(full) = self
            .open
            .take_if(|segment| segment.len() == segment.capacity())
        {
            if self.full.is_empty() && full.len() < MAX_FRAME_SIZE {
                // Grown rather than closed: a message that fits in a frame is read as one segment.
                let mut grown = BUFFERS.take((full.len() + wanted).min(MAX_FRAME_SIZE))?;
                grown.extend_from_slice(&full);
                self.open = Some(grown);
            } else {
                self.full.push(full.into_bytes());
            }
        }
        let capacity = self.next_capacity(wanted);
        match &mut self.open {
            Some(segment) => Ok(segment),
            open => {
                let mut segment = BUFFERS.take(capacity)?;
                segment.append(&mut self.small);
                Ok(open.insert(segment))
            }
        }
    }

    /// The first segment is sized to the bytes in hand, so a message just past the whole size does
    /// not pin a whole frame until it is acknowledged; every later one is a frame.
    fn next_capacity(&self, wanted: usize) -> usize {
        if self.full.is_empty() {
            (self.small.len() + wanted).min(MAX_FRAME_SIZE)
        } else {
            MAX_FRAME_SIZE
        }
    }

    fn is_small_after(&self, more: usize) -> bool {
        self.open.is_none()
            && self.full.is_empty()
            && self.small.len() + more <= MAX_WHOLE_MESSAGE_SIZE
    }
}

impl Write for SegmentWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.is_small_after(buf.len()) {
            self.small
                .try_reserve(buf.len())
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
            self.small.extend_from_slice(buf);
            return Ok(buf.len());
        }
        let segment = self
            .open_segment(buf.len())
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        let room = buf.len().min(segment.capacity() - segment.len());
        segment.extend_from_slice(&buf[..room]);
        Ok(room)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::TensorData;

    #[derive(Serialize)]
    struct Upload(TensorData);

    impl Encode for Upload {}

    /// Raw bytes as if they were a message's encoding, to test how messages are framed.
    impl From<&[u8]> for Encoded {
        fn from(bytes: &[u8]) -> Self {
            let mut writer = SegmentWriter::default();
            writer.write_all(bytes).unwrap();
            writer.finish()
        }
    }

    #[test]
    fn a_small_message_is_one_segment() {
        let upload = Upload(TensorData::new(vec![1.0f32; 16], [16]));
        let expected = rmp_serde::to_vec(&upload).unwrap();

        let segments: Vec<_> = upload.encode().unwrap().into_segments().collect();

        assert_eq!(segments, [expected]);
    }

    #[test]
    fn a_large_message_is_encoded_in_segments_of_one_frame() {
        let values = 5 * MAX_FRAME_SIZE / size_of::<f32>() / 2;
        let upload = Upload(TensorData::new(vec![1.0f32; values], [values]));
        let expected = rmp_serde::to_vec(&upload).unwrap();

        let encoded = upload.encode().unwrap();

        assert_eq!(encoded.len(), expected.len());
        let segments: Vec<_> = encoded.into_segments().collect();
        let (last, full) = segments.split_last().unwrap();
        assert_eq!(full.len(), 2);
        assert!(full.iter().all(|segment| segment.len() == MAX_FRAME_SIZE));
        assert!(!last.is_empty() && last.len() <= MAX_FRAME_SIZE);
        assert_eq!(segments.concat(), expected);
    }

    #[test]
    fn a_message_of_exactly_one_frame_ends_without_an_empty_segment() {
        let encoded = Encoded::from(&vec![7; MAX_FRAME_SIZE][..]);

        let segments: Vec<_> = encoded.into_segments().collect();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].len(), MAX_FRAME_SIZE);
    }
}
