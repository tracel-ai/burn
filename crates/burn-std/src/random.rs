use crate::rand::{RngExt, SeedableRng, StdRng};
use serde::{Deserialize, Serialize};

/// Explicit randomness for one operation.
///
/// Sampling leaves the key unchanged and does not advance the backend's implicit generator.
/// Reusing a key repeats a draw under the same execution conditions. Use separate keys for
/// separate draws, either allocated by the backend or derived with [`split`](Self::split).
/// Identical values across backends, sharding layouts, or library versions are not guaranteed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RandomKey {
    seed: u64,
}

impl RandomKey {
    /// Creates an operation key from a seed, without changing any backend's generator.
    pub const fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Returns the seed used to initialize an operation-local generator.
    ///
    /// Intended for backend implementations and IR encoding. Use [`split`](Self::split)
    /// to derive keys rather than performing arithmetic on this seed.
    pub const fn seed(self) -> u64 {
        self.seed
    }

    /// Derives an infinite sequence of child keys in a deterministic order.
    ///
    /// Each call starts the same sequence; take as many children as the operation needs.
    /// Derivation does not advance the backend's implicit generator.
    pub fn split(self) -> impl Iterator<Item = Self> {
        let mut rng = StdRng::seed_from_u64(self.seed());
        core::iter::from_fn(move || Some(Self::new(rng.random())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitting_is_repeatable_and_separates_children() {
        let key = RandomKey::new(7);
        let children: alloc::vec::Vec<_> = key.split().take(2).collect();
        assert_eq!(
            children,
            key.split().take(2).collect::<alloc::vec::Vec<_>>()
        );
        assert_ne!(
            children,
            RandomKey::new(19)
                .split()
                .take(2)
                .collect::<alloc::vec::Vec<_>>()
        );
        assert_ne!(children[0], children[1]);
        assert!(!children.contains(&key));
    }
}
