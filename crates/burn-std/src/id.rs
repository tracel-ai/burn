//! # Unique Identifiers
use crate::rand::gen_random;

/// Simple ID generator.
pub struct IdGenerator {}

impl IdGenerator {
    /// Generates a new ID.
    pub fn generate() -> u64 {
        // Generate a random u64 (18,446,744,073,709,551,615 combinations)
        let random_bytes: [u8; 8] = gen_random();
        u64::from_le_bytes(random_bytes)
    }
}

pub use cubecl_common::stream_id::StreamId;

use core::hash::{BuildHasher, Hasher};

use alloc::str::FromStr;
use data_encoding::BASE32_DNSSEC;
use serde::{Deserialize, Serialize};

// Hashbrown changed its default hasher in 0.15, but there are some issues
// https://github.com/rust-lang/hashbrown/issues/577
// Also, `param_serde_deserialize_legacy_uuid` doesn't pass with the default hasher.
type DefaultHashBuilder = core::hash::BuildHasherDefault<ahash::AHasher>;

/// Unique ID for a parameter of a module.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Serialize, Deserialize)]
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
}

impl FromStr for ParamId {
    type Err = &'static str; // Or a custom error type if preferred

    /// Construct a param id from str.
    ///
    /// Preserves compatibility with previous formats (6 bytes, 16-byte uuid).
    ///
    /// # Returns
    /// A `Result` containing the `ParamId` when valid.
    fn from_str(encoded: &str) -> Result<Self, Self::Err> {
        let u64_id: Option<u64> = match BASE32_DNSSEC.decode(encoded.as_bytes()) {
            Ok(bytes) => {
                let mut buffer = [0u8; 8];
                buffer[..bytes.len()].copy_from_slice(&bytes);
                Some(u64::from_le_bytes(buffer))
            }
            Err(_) => match uuid::Uuid::try_parse(encoded) {
                // Backward compatibility with uuid parameter identifiers
                Ok(id) => {
                    // Hash the 128-bit uuid to 64-bit
                    // Though not *theoretically* unique, the probability of a collision should be extremely low
                    let mut hasher = DefaultHashBuilder::default().build_hasher();
                    // let mut hasher = DefaultHasher::new();
                    hasher.write(id.as_bytes());
                    Some(hasher.finish())
                }
                Err(_) => None,
            },
        };
        u64_id.map(Self::from).ok_or("Invalid id.")
    }
}

impl core::fmt::Display for ParamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let encoded = BASE32_DNSSEC.encode(&self.value.to_le_bytes());
        f.write_str(&encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::collections::BTreeSet;
    use alloc::string::ToString;

    #[cfg(feature = "std")]
    use dashmap::DashSet; //Concurrent HashMap
    #[cfg(feature = "std")]
    use std::{sync::Arc, thread};

    #[test]
    fn uniqueness_test() {
        const IDS_CNT: usize = 10_000;

        let mut set: BTreeSet<u64> = BTreeSet::new();

        for _i in 0..IDS_CNT {
            assert!(set.insert(IdGenerator::generate()));
        }

        assert_eq!(set.len(), IDS_CNT);
    }

    #[cfg(feature = "std")]
    #[test]
    fn thread_safety_test() {
        const NUM_THREADS: usize = 10;
        const NUM_REPEATS: usize = 1_000;
        const EXPECTED_TOTAL_IDS: usize = NUM_THREADS * NUM_REPEATS;

        let set: Arc<DashSet<u64>> = Arc::new(DashSet::new());

        let mut handles = vec![];

        for _ in 0..NUM_THREADS {
            let set = set.clone();

            let handle = thread::spawn(move || {
                for _i in 0..NUM_REPEATS {
                    assert!(set.insert(IdGenerator::generate()));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }
        assert_eq!(set.len(), EXPECTED_TOTAL_IDS);
    }

    #[test]
    fn param_serde_try_deserialize() {
        let val = ParamId::from(123456u64);
        let deserialized = ParamId::from_str(&val.to_string()).unwrap();
        assert_eq!(val, deserialized);

        assert_eq!(ParamId::from_str("invalid_id"), Err("Invalid id."));
    }

    #[test]
    fn param_serde_deserialize() {
        let val = ParamId::from(123456u64);
        let deserialized = ParamId::from_str(&val.to_string()).unwrap();
        assert_eq!(val, deserialized);
    }

    #[test]
    fn param_serde_deserialize_legacy() {
        let legacy_val = [45u8; 6];
        let param_id = ParamId::from_str(&BASE32_DNSSEC.encode(&legacy_val)).unwrap();
        assert_eq!(param_id.val().to_le_bytes()[0..6], legacy_val);
        assert_eq!(param_id.val().to_le_bytes()[6..], [0, 0]);
    }

    #[test]
    fn param_serde_deserialize_legacy_uuid() {
        // Ensure support for legacy uuid deserialization and make sure it results in the same output
        let legacy_id = "30b82c23-788d-4d63-a743-ada258d5f13c";
        let param_id1 = ParamId::from_str(legacy_id).unwrap();
        let param_id2 = ParamId::from_str(legacy_id).unwrap();
        assert_eq!(param_id1, param_id2);
    }

    #[test]
    #[should_panic = "Invalid id."]
    fn param_serde_deserialize_invalid_id() {
        let invalid_uuid = "30b82c23-788d-4d63-ada258d5f13c";
        let _ = ParamId::from_str(invalid_uuid).unwrap();
    }
}
