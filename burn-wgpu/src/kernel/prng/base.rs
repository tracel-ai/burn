use burn_common::rand::get_seeded_rng;
use burn_tensor::{Distribution, Shape};
use rand::Rng;

use crate::{
    element::WgpuElement,
    kernel::{elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    pool::get_context,
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice, SEED,
};

kernel_wgsl!(PRNG, "../../template/prng/default copy.wgsl");

pub fn random<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    distribution: Distribution<E>,
    device: &WgpuDevice,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let mut rng = match SEED.lock().unwrap().as_ref() {
        Some(rng_seeded) => rng_seeded.clone(),
        None => get_seeded_rng(),
    };
    let mut seeds: Vec<u32> = Vec::with_capacity(4);
    for _ in 0..4 {
        seeds.push(rng.gen());
    }

    let context = get_context::<G>(device);
    let num_elems = shape.num_elements();
    let buffer = context.create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(context.clone(), shape, buffer);

    let kernel = context.compile_static::<KernelSettings<PRNG, E, i32, WORKGROUP, WORKGROUP, 1>>();

    let info_buffer = context.create_buffer_with_data(bytemuck::cast_slice(&seeds));

    context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&output.buffer, &info_buffer],
    );

    // For now, only default distribution (uniform in [0,1[)
    // Then:
    // match distribution {
    //     Distribution::Default => keep,
    //     Distribution::Bernoulli(prob) => greater,
    //     Distribution::Uniform(from, to) => stretch,
    //     Distribution::Normal(mean, std) => box-muller
    // }
    // transform default to distribution
    // replace by each kernel later

    // ?
    // *seed = Some(rng);

    output
}

#[cfg(test)]
mod tests {
    use burn_tensor::{Distribution, Shape, Tensor};
    use num_traits::{Float, FloatConst};

    use crate::{
        tests::{ReferenceBackend, TestBackend},
        WgpuDevice,
    };

    #[test]
    fn subsequent_calls_give_different_tensors() {
        let shape = [4, 5];
        let device = WgpuDevice::default();

        let tensor_1 =
            Tensor::<TestBackend, 2>::random_device(shape.clone(), Distribution::Default, &device);
        let tensor_2 =
            Tensor::<TestBackend, 2>::random_device(shape.clone(), Distribution::Default, &device);
        for i in 0..20 {
            assert!(tensor_1.to_data().value[i] != tensor_2.to_data().value[i]);
        }
    }

    #[test]
    fn runs_test() {
        let shape = Shape::new([1000, 1000]);
        let device = WgpuDevice::default();
        let tensor =
            Tensor::<TestBackend, 2>::random_device(shape.clone(), Distribution::Default, &device);
        // let tensor = Tensor::<ReferenceBackend, 2>::random(shape.clone(), Distribution::Default);
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
        println!("{:?}", z);
        assert!(false);
        // assert!(z.abs() < 3.29); // 99.9% confidence
    }
}
