use crate::rand::gen_random;
use serde::{Deserialize, Serialize};

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

/// Unique identifier that can represent a stream based on the current thread id.
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub struct StreamId {
    /// The value representing the thread id.
    pub value: u64,
}

impl StreamId {
    /// Get the current thread id.
    pub fn current() -> Self {
        Self {
            #[cfg(feature = "std")]
            value: Self::from_current_thread(),
            #[cfg(not(feature = "std"))]
            value: 0,
        }
    }

    #[cfg(feature = "std")]
    fn from_current_thread() -> u64 {
        use core::hash::Hash;

        std::thread_local! {
            static ID: std::cell::OnceCell::<u64> = const { std::cell::OnceCell::new() };
        };

        // Getting the current thread is expensive, so we cache the value into a thread local
        // variable, which is very fast.
        ID.with(|cell| {
            *cell.get_or_init(|| {
                // A way to get a thread id encoded as u64.
                let mut hasher = std::hash::DefaultHasher::default();
                let id = std::thread::current().id();
                id.hash(&mut hasher);
                std::hash::Hasher::finish(&hasher)
            })
        })
    }
}

impl core::fmt::Display for StreamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("StreamId({:?})", self.value))
    }
}
