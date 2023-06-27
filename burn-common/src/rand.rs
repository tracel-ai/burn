pub use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(feature = "std")]
use std::sync::Mutex;

#[cfg(not(feature = "std"))]
use crate::stub::Mutex;

#[cfg(not(feature = "std"))]
use const_random::const_random;

/// Returns a seeded random number generator using entropy.
#[cfg(feature = "std")]
#[inline(always)]
pub fn get_seeded_rng() -> StdRng {
    StdRng::from_entropy()
}

/// Returns a seeded random number generator using a pre-generated seed.
#[cfg(not(feature = "std"))]
#[inline(always)]
pub fn get_seeded_rng() -> StdRng {
    const GENERATED_SEED: u64 = const_random!(u64);
    StdRng::seed_from_u64(GENERATED_SEED)
}

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);
