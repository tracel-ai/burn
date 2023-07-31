use burn_common::rand::get_seeded_rng;
use burn_tensor::Shape;
use rand::Rng;

use crate::{
    context::WorkGroup, element::WgpuElement, kernel::KernelSettings, kernel_wgsl,
    pool::get_context, tensor::WgpuTensor, GraphicsApi, WgpuDevice, SEED,
};

kernel_wgsl!(PRNG, "../../template/prng/default.wgsl");

fn get_seeds() -> Vec<u32> {
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

/// Pseudo-random generator for default distribution (uniform in [0,1[)
pub fn random_default<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;
    const N_VALUES_PER_THREAD: u32 = 100;
    let num_elems = shape.num_elements();
    let num_threads = f32::ceil(num_elems as f32 / N_VALUES_PER_THREAD as f32);
    let num_invocations = f32::ceil(num_threads / (WORKGROUP * WORKGROUP) as f32);
    let workgroup_x = f32::ceil(f32::sqrt(num_invocations));
    let workgroup_y = f32::ceil(num_invocations / workgroup_x);
    let workgroup = WorkGroup::new(workgroup_x as u32, workgroup_y as u32, 1);

    let context = get_context::<G>(device);
    let buffer = context.create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(context.clone(), shape, buffer);

    let kernel = context.compile_static::<KernelSettings<PRNG, E, i32, WORKGROUP, WORKGROUP, 1>>();

    let mut info = get_seeds();
    info.insert(0, N_VALUES_PER_THREAD);
    let info_buffer = context.create_buffer_with_data(bytemuck::cast_slice(&info));

    context.execute(workgroup, kernel, &[&output.buffer, &info_buffer]);

    output
}

#[cfg(test)]
mod tests {
    use burn_tensor::{backend::Backend, Distribution, Shape, Tensor};
    use num_traits::Float;

    use crate::{tests::TestBackend, WgpuDevice};

    #[test]
    fn subsequent_calls_give_different_tensors() {
        TestBackend::seed(0);
        let shape = [4, 5];
        let device = WgpuDevice::default();

        let tensor_1 =
            Tensor::<TestBackend, 2>::random_device(shape, Distribution::Default, &device);
        let tensor_2 =
            Tensor::<TestBackend, 2>::random_device(shape, Distribution::Default, &device);
        for i in 0..20 {
            assert!(tensor_1.to_data().value[i] != tensor_2.to_data().value[i]);
        }
    }

    #[test]
    fn runs_test() {
        TestBackend::seed(0);
        let shape = Shape::new([500, 500]);
        let device = WgpuDevice::default();
        let tensor =
            Tensor::<TestBackend, 2>::random_device(shape.clone(), Distribution::Default, &device);

        let numbers = tensor.clone().into_data().value;
        let mut n_runs = 1;
        let mut n_0 = 0.;
        let mut n_1 = 0.;
        for i in 1..numbers.len() {
            let bin = numbers[i] < 0.5;
            match bin {
                true => n_0 += 1.,
                false => n_1 += 1.,
            };
            let last_bin = numbers[i - 1] < 0.5;
            if bin != last_bin {
                n_runs += 1;
            }
        }

        let expectation = (2. * n_0 * n_1) / (n_0 + n_1) + 1.0;
        let variance = ((2. * n_0 * n_1) * (2. * n_0 * n_1 - n_0 - n_1))
            / ((n_0 + n_1).powf(2.) * (n_0 + n_1 - 1.));
        let z = (n_runs as f32 - expectation) / variance.sqrt();

        // below 1.96 means we can have good confidence in the randomness
        assert!(z.abs() < 1.96);
    }
}
