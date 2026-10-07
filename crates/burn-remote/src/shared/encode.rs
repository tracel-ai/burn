use std::io::{self, Write};

use rmp_serde::encode::Error;
use serde::Serialize;

/// Below this much tensor data, the copy a growing buffer makes costs less than the pass that
/// sizes the buffer instead.
const SIZED_ENCODING_MIN: usize = 64 * 1024;

/// A message sent to a peer: a buffer grown while encoding is left full after a tensor's bytes,
/// and the next field doubles it, copying the tensor again.
pub trait WireMessage: Serialize {
    /// Size of the tensor data the message carries, in bytes.
    fn data_len(&self) -> usize;

    /// The message's MessagePack bytes, in a buffer of exactly their size when the message carries
    /// enough tensor data for that to save a copy.
    fn encode(&self) -> Result<Vec<u8>, Error> {
        if self.data_len() < SIZED_ENCODING_MIN {
            return rmp_serde::to_vec(self);
        }
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(EncodedLen::of(self)?).is_err() {
            // `to_vec` reports the failed allocation as rmp does, an out-of-memory write.
            return rmp_serde::to_vec(self);
        }
        rmp_serde::encode::write(&mut bytes, self)?;
        Ok(bytes)
    }
}

/// The length of a value's MessagePack encoding, counted without keeping the bytes.
pub struct EncodedLen(usize);

impl EncodedLen {
    /// Count the encoding of `value`.
    pub fn of<T: Serialize + ?Sized>(value: &T) -> Result<usize, Error> {
        let mut len = Self(0);
        rmp_serde::encode::write(&mut len, value)?;
        Ok(len.0)
    }
}

impl Write for EncodedLen {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0 += buf.len();
        Ok(buf.len())
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

    impl WireMessage for Upload {
        fn data_len(&self) -> usize {
            self.0.bytes().len()
        }
    }

    #[test]
    fn a_large_tensor_is_encoded_into_a_buffer_of_exactly_its_size() {
        let upload = Upload(TensorData::new(vec![1.0f32; 64 * 1024], [64 * 1024]));

        let bytes = upload.encode().unwrap();

        assert_eq!(bytes, rmp_serde::to_vec(&upload).unwrap());
        assert_eq!(bytes.capacity(), bytes.len());
    }

    #[test]
    fn an_encoded_len_matches_the_encoding() {
        let upload = Upload(TensorData::new(vec![1.0f32; 16], [16]));

        assert_eq!(
            EncodedLen::of(&upload).unwrap(),
            rmp_serde::to_vec(&upload).unwrap().len()
        );
    }
}
