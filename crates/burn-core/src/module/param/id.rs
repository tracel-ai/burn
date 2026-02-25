use core::hash::{BuildHasher, Hasher};

use alloc::string::String;
use burn_std::id::IdGenerator;
use data_encoding::BASE32_DNSSEC;

// Hashbrown changed its default hasher in 0.15, but there are some issues
// https://github.com/rust-lang/hashbrown/issues/577
// Also, `param_serde_deserialize_legacy_uuid` doesn't pass with the default hasher.
type DefaultHashBuilder = core::hash::BuildHasherDefault<ahash::AHasher>;

/// Unique ID for a parameter of a burn module.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord)]
pub struct ParamId {
    value: u64,
}

impl From<u64> for ParamId {
    fn from(value: u64) -> Self {
        Self { value }
    }
}

impl Default for ParamId {
    fn default() -> Self {
        Self::new()
    }
}

impl ParamId {
    /// Create a new parameter ID.
    pub fn new() -> Self {
        Self {
            value: IdGenerator::generate(),
        }
    }

    /// Gets the internal value of the id.
    pub fn val(&self) -> u64 {
        self.value
    }

    /// Convert the parameter ID into a string.
    pub fn serialize(self) -> String {
        BASE32_DNSSEC.encode(&self.value.to_le_bytes())
    }

    /// Deserialize a param id.
    ///
    /// Preserves compatibility with previous formats (6 bytes, 16-byte uuid).
    pub fn deserialize(encoded: &str) -> ParamId {
        let u64_id = match BASE32_DNSSEC.decode(encoded.as_bytes()) {
            Ok(bytes) => {
                let mut buffer = [0u8; 8];
                buffer[..bytes.len()].copy_from_slice(&bytes);
                u64::from_le_bytes(buffer)
            }
            Err(err) => match uuid::Uuid::try_parse(encoded) {
                // Backward compatibility with uuid parameter identifiers
                Ok(id) => {
                    // Hash the 128-bit uuid to 64-bit
                    // Though not *theoretically* unique, the probability of a collision should be extremely low
                    let mut hasher = DefaultHashBuilder::default().build_hasher();
                    // let mut hasher = DefaultHasher::new();
                    hasher.write(id.as_bytes());
                    hasher.finish()
                }
                Err(_) => panic!("Invalid id. {err}"),
            },
        };

        ParamId::from(u64_id)
    }
}

impl core::fmt::Display for ParamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.serialize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_serde_deserialize() {
        let val = ParamId::from(123456u64);
        let deserialized = ParamId::deserialize(&val.serialize());
        assert_eq!(val, deserialized);
    }

    #[test]
    fn param_serde_deserialize_legacy() {
        let legacy_val = [45u8; 6];
        let param_id = ParamId::deserialize(&BASE32_DNSSEC.encode(&legacy_val));
        assert_eq!(param_id.val().to_le_bytes()[0..6], legacy_val);
        assert_eq!(param_id.val().to_le_bytes()[6..], [0, 0]);
    }

    #[test]
    fn param_serde_deserialize_legacy_uuid() {
        // Ensure support for legacy uuid deserialization and make sure it results in the same output
        let legacy_id = "30b82c23-788d-4d63-a743-ada258d5f13c";
        let param_id1 = ParamId::deserialize(legacy_id);
        let param_id2 = ParamId::deserialize(legacy_id);
        assert_eq!(param_id1, param_id2);
    }

    #[test]
    #[should_panic = "Invalid id."]
    fn param_serde_deserialize_invalid_id() {
        let invalid_uuid = "30b82c23-788d-4d63-ada258d5f13c";
        let _ = ParamId::deserialize(invalid_uuid);
    }
}
