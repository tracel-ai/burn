pub use rand::{rngs::StdRng, Rng, SeedableRng};

#[cfg(not(feature = "std"))]
use const_random::const_random;
use rand::distributions::Standard;
use rand::prelude::Distribution;

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

/// Generates random data from a thread-local RNG.
#[cfg(feature = "std")]
#[inline]
pub fn gen_random<T>() -> T
where
    Standard: Distribution<T>,
{
    rand::thread_rng().gen()
}

/// Generates random data from a mutex-protected RNG.
#[cfg(not(feature = "std"))]
#[inline]
pub fn gen_random<T>() -> T
where
    Standard: Distribution<T>,
{
    use crate::stub::Mutex;
    static RNG: Mutex<Option<StdRng>> = Mutex::new(None);
    let mut rng = RNG.lock().unwrap();
    if rng.is_none() {
        *rng = Some(get_seeded_rng());
    }
    rng.as_mut().unwrap().gen()
}
