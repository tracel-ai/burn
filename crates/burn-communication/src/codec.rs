//! Wire codec shared by the communication and remote backends.
//!
//! Messages are encoded with CBOR (via `ciborium`), the same serde binary codec
//! used by `burn-pack`, so the project depends on a single serialization crate.

use serde::{Serialize, de::DeserializeOwned};

/// Serialize a value to CBOR bytes.
///
/// # Panics
/// Panics if the value cannot be serialized, which only happens for types whose
/// `Serialize` implementation can fail (none of the wire message types can).
pub fn serialize<T: Serialize + ?Sized>(value: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf).expect("CBOR serialization failed");
    buf
}

/// Deserialize a value from CBOR bytes.
pub fn deserialize<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    ciborium::de::from_reader(bytes).map_err(|err| err.to_string())
}
