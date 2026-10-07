use rmp_serde::encode::Error;
use serde::Serialize;

/// A message's MessagePack encoding, as sent to a peer.
pub trait Encode: Serialize + Sized {
    /// The message's MessagePack bytes. Takes the message so its tensor data is freed before the
    /// bytes go out, not after.
    fn encode(self) -> Result<Vec<u8>, Error> {
        rmp_serde::to_vec(&self)
    }
}
