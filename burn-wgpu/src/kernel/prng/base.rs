use burn_common::rand::get_seeded_rng;
use rand::Rng;

use crate::SEED;

pub(crate) fn get_seeds() -> Vec<u32> {
    let mut seed = SEED.lock().unwrap();
    let mut rng = match seed.as_ref() {
        Some(rng_seeded) => rng_seeded.clone(),
        None => get_seeded_rng(),
    };
    let mut seeds: Vec<u32> = Vec::with_capacity(4);
    for _ in 0..4 {
        seeds.push(rng.gen());
    }
    *seed = Some(rng);
    seeds
}
