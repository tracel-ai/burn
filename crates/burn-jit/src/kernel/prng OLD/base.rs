use crate::{element::JitElement, kernel_wgsl, Runtime, SEED};
use burn_common::rand::get_seeded_rng;
use burn_compute::{client::ComputeClient, server::Handle};
use rand::Rng;

kernel_wgsl!(Prng, "../../template/prng/prng.wgsl");

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

pub(crate) fn make_info_buffer<R: Runtime>(
    client: ComputeClient<R::Server, R::Channel>,
    n_values_per_thread: usize,
) -> Handle<R::Server> {
    let mut info = get_seeds();
    info.insert(0, n_values_per_thread as u32);
    client.create(bytemuck::cast_slice(&info))
}

pub(crate) fn make_args_buffer<R: Runtime, E: JitElement>(
    client: ComputeClient<R::Server, R::Channel>,
    args: &[E],
) -> Handle<R::Server> {
    client.create(E::as_bytes(args))
}

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
pub mod tests_utils {
    use burn_tensor::Element;

    #[derive(Default, Copy, Clone)]
    pub struct BinStats {
        pub count: usize,
        pub n_runs: usize, // Number of sequences of same bin
    }

    pub fn calculate_bin_stats<E: Element>(
        numbers: Vec<E>,
        number_of_bins: usize,
        low: f32,
        high: f32,
    ) -> Vec<BinStats> {
        let range = (high - low) / number_of_bins as f32;
        let mut output: Vec<BinStats> = (0..number_of_bins).map(|_| Default::default()).collect();
        let mut initialized = false;
        let mut current_runs = number_of_bins; // impossible value for starting point
        for number in numbers {
            let num = number.elem::<f32>();
            if num < low || num > high {
                continue;
            }
            let index = f32::floor((num - low) / range) as usize;
            output[index].count += 1;
            if initialized && index != current_runs {
                output[current_runs].n_runs += 1;
            }
            initialized = true;
            current_runs = index;
        }
        output[current_runs].n_runs += 1;
        output
    }
}
