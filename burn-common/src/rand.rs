use crate::stub::Mutex;

use rand::{rngs::StdRng, SeedableRng};

use const_random::const_random;

#[inline(always)]
pub fn get_seeded_rng() -> StdRng {
    const GENERATED_SEED: u64 = const_random!(u64);
    StdRng::seed_from_u64(GENERATED_SEED)
}

pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);
