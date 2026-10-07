use std::{
    io::{self, Write},
    mem,
};

use bytes::Bytes;
use rmp_serde::encode::Error;
use serde::Serialize;

use crate::transport::link::MAX_FRAME_SIZE;

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

/// A message's bytes in segments of at most one frame, each its own allocation, so each frame can
/// be freed once it is sent.
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

/// Raw bytes as if they were a message's encoding, to test how messages are framed.
#[cfg(test)]
impl From<&[u8]> for Encoded {
    fn from(bytes: &[u8]) -> Self {
        let mut writer = SegmentWriter::default();
        writer.write_all(bytes).unwrap();
        writer.finish()
    }
}

/// Collects an encoding into segments: the first grows up to a frame, and each later one is
/// allocated a full frame up front, so no byte past the first frame is copied to grow a buffer.
#[derive(Default)]
struct SegmentWriter {
    full: Vec<Bytes>,
    open: Vec<u8>,
    len: usize,
}

impl SegmentWriter {
    fn finish(mut self) -> Encoded {
        if !self.open.is_empty() {
            self.full.push(self.open.into());
        }
        Encoded {
            segments: self.full,
            len: self.len,
        }
    }

    /// Room in the open segment for some of `wanted` more bytes, closing it first if it is full.
    fn make_room(&mut self, wanted: usize) -> io::Result<usize> {
        if self.open.len() == MAX_FRAME_SIZE {
            self.full.push(mem::take(&mut self.open).into());
        }
        let capacity = if self.full.is_empty() {
            (self.open.len() + wanted)
                .max(2 * self.open.capacity())
                .min(MAX_FRAME_SIZE)
        } else {
            MAX_FRAME_SIZE
        };
        if capacity > self.open.capacity() {
            self.open
                .try_reserve_exact(capacity - self.open.len())
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        }
        Ok(self.open.capacity().min(MAX_FRAME_SIZE) - self.open.len())
    }
}

impl Write for SegmentWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let written = buf.len().min(self.make_room(buf.len())?);
        self.open.extend_from_slice(&buf[..written]);
        self.len += written;
        Ok(written)
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
