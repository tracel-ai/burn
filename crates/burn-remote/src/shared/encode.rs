use std::io::{self, Write};

use rmp_serde::encode::Error;
use serde::Serialize;

/// Encodes a message into a buffer of exactly its size: a buffer grown while encoding is left full
/// after a tensor's bytes, and the next field doubles it, copying the tensor again.
pub trait EncodeExact: Serialize {
    /// The message's MessagePack bytes, in a buffer of exactly their length.
    fn encode_exact(&self) -> Result<Vec<u8>, Error> {
        let mut size = EncodedSize(0);
        rmp_serde::encode::write(&mut size, self)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size.0)
            .map_err(|err| Error::Syntax(format!("cannot allocate {} bytes: {err}", size.0)))?;
        rmp_serde::encode::write(&mut bytes, self)?;
        Ok(bytes)
    }
}

impl<T: Serialize + ?Sized> EncodeExact for T {}

/// Counts what an encoder writes without keeping it.
struct EncodedSize(usize);

impl Write for EncodedSize {
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

    #[test]
    fn a_tensor_is_encoded_into_a_buffer_of_exactly_its_size() {
        let data = TensorData::new(vec![1.0f32; 1024], [1024]);

        let bytes = data.encode_exact().unwrap();

        assert_eq!(bytes, rmp_serde::to_vec(&data).unwrap());
        assert_eq!(bytes.capacity(), bytes.len());
    }
}
