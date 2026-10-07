use std::io::{self, Write};

use rmp_serde::encode::Error;
use serde::Serialize;

/// Below this much tensor data, the copy a growing buffer makes costs less than the pass that
/// sizes the buffer instead.
const SIZED_ENCODING_MIN: usize = 64 * 1024;

/// A message's MessagePack encoding, as sent to a peer.
pub trait Encode: Serialize {
    /// Size of the tensor data the message carries, in bytes, which decides whether `encode` sizes
    /// its buffer first.
    fn data_len(&self) -> usize;

    /// Length of the message's MessagePack encoding, counted without keeping the bytes.
    fn encoded_len(&self) -> Result<usize, Error> {
        let mut counter = ByteCounter(0);
        rmp_serde::encode::write(&mut counter, self)?;
        Ok(counter.0)
    }

    /// The message's MessagePack bytes. Takes the message so its tensor data is freed before the
    /// bytes go out, not after.
    fn encode(self) -> Result<Vec<u8>, Error>
    where
        Self: Sized,
    {
        if self.data_len() < SIZED_ENCODING_MIN {
            return rmp_serde::to_vec(&self);
        }
        // A growing buffer ends full after a tensor's bytes, and the next field doubles it.
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(self.encoded_len()?).is_err() {
            // Only rmp can build its out-of-memory error, and `to_vec` returns it.
            return rmp_serde::to_vec(&self);
        }
        rmp_serde::encode::write(&mut bytes, &self)?;
        Ok(bytes)
    }
}

struct ByteCounter(usize);

impl Write for ByteCounter {
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

    impl Encode for Upload {
        fn data_len(&self) -> usize {
            self.0.bytes().len()
        }
    }

    #[test]
    fn a_large_tensor_is_encoded_into_a_buffer_of_exactly_its_size() {
        let upload = Upload(TensorData::new(vec![1.0f32; 64 * 1024], [64 * 1024]));
        let expected = rmp_serde::to_vec(&upload).unwrap();

        let bytes = upload.encode().unwrap();

        assert_eq!(bytes, expected);
        assert_eq!(bytes.capacity(), bytes.len());
    }

    #[test]
    fn an_encoded_len_matches_the_encoding() {
        let upload = Upload(TensorData::new(vec![1.0f32; 16], [16]));

        assert_eq!(
            upload.encoded_len().unwrap(),
            rmp_serde::to_vec(&upload).unwrap().len()
        );
    }
}
