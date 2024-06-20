use alloc::string::String;

use crate::rand::gen_random;

use data_encoding::BASE32_DNSSEC;

/// Simple ID generator.
pub struct IdGenerator {}

impl IdGenerator {
    /// Generates a new ID.
    pub fn generate() -> String {
        // Generate 6 random bytes (281,474,976,710,656 combinations)
        let random_bytes: [u8; 6] = gen_random();

        // Encode the random bytes in base32 DNSSEC
        // 6 bytes encodes to 10 lower case characters, e.g. "3uu5e6vv7c"
        BASE32_DNSSEC.encode(&random_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::{collections::BTreeSet, string::String};

    #[cfg(feature = "std")]
    use dashmap::DashSet; //Concurrent HashMap
    #[cfg(feature = "std")]
    use std::{sync::Arc, thread};

    #[test]
    fn not_empty_test() {
        assert!(!IdGenerator::generate().is_empty());
    }

    #[test]
    fn uniqueness_test() {
        const IDS_CNT: usize = 10_000;

        let mut set: BTreeSet<String> = BTreeSet::new();

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

        let set: Arc<DashSet<String>> = Arc::new(DashSet::new());

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
}
