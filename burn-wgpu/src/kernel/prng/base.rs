use std::sync::Arc;

use burn_common::rand::get_seeded_rng;
use burn_tensor::Shape;
use rand::Rng;
use wgpu::Buffer;

use crate::{context::Context, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor, SEED};

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

pub(crate) fn make_output_tensor<E: WgpuElement, const D: usize>(
    context: Arc<Context>,
    shape: Shape<D>,
) -> WgpuTensor<E, D> {
    let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<E>());
    WgpuTensor::new(context, shape, buffer)
}

pub(crate) fn make_info_buffer(context: Arc<Context>, n_values_per_thread: usize) -> Arc<Buffer> {
    let mut info = get_seeds();
    info.insert(0, n_values_per_thread as u32);
    context.create_buffer_with_data(bytemuck::cast_slice(&info))
}

pub(crate) fn make_args_buffer<E: WgpuElement>(context: Arc<Context>, args: &[E]) -> Arc<Buffer> {
    context.create_buffer_with_data(E::as_bytes(args))
}

#[cfg(test)]
pub mod tests {
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

    #[test]
    fn test_count_bins() {
        let numbers = vec![0., 1., 1.5, 2., 2.5, 3., 2.5, 1.5, 3.5];
        let number_of_bins = 4;
        let low = 0.;
        let high = 4.;
        let stats = calculate_bin_stats(numbers, number_of_bins, low, high);
        assert_eq!(stats[0].count, 1);
        assert_eq!(stats[0].n_runs, 1);
        assert_eq!(stats[1].count, 3);
        assert_eq!(stats[1].n_runs, 2);
        assert_eq!(stats[2].count, 3);
        assert_eq!(stats[2].n_runs, 2);
        assert_eq!(stats[3].count, 2);
        assert_eq!(stats[3].n_runs, 2);
    }
}
