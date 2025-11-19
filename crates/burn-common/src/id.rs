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

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::collections::BTreeSet;

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
}
