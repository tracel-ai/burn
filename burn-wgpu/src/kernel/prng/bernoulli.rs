use burn_tensor::Shape;

use crate::{
    compute::{compute_client, StaticKernel},
    element::WgpuElement,
    kernel::{
        prng::base::{make_args_buffer, make_info_buffer},
        prng_workgroup, KernelSettings, SourceTemplate, StaticKernelSource,
    },
    ops::numeric::empty_device,
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice,
};

use super::base::Prng;

struct BernoulliPrng;

impl StaticKernelSource for BernoulliPrng {
    fn source() -> SourceTemplate {
        Prng::source()
            .register("num_args", "1")
            .register(
                "prng_loop",
                include_str!("../../template/prng/bernoulli_inner_loop.wgsl"),
            )
            .add_template("fn cast_elem(e: bool) -> {{ elem }} {return {{elem}}(e);}")
    }
}

/// Pseudo-random generator for bernoulli
pub fn random_bernoulli<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
    prob: E,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;
    const N_VALUES_PER_THREAD: usize = 128;

    let client = compute_client::<G>(device);
    let output = empty_device(client.clone(), device.clone(), shape.clone());
    let info_handle = make_info_buffer(client.clone(), N_VALUES_PER_THREAD);
    let args_handle = make_args_buffer(client.clone(), &[prob]);
    let workgroup = prng_workgroup(shape.num_elements(), WORKGROUP, N_VALUES_PER_THREAD);
    let kernel =
        StaticKernel::<KernelSettings<BernoulliPrng, E, i32, WORKGROUP, WORKGROUP, 1>>::new(
            workgroup,
        );

    client.execute(
        Box::new(kernel),
        &[&output.handle, &info_handle, &args_handle],
    );

    output
}

#[cfg(test)]
mod tests {
    use core::f32;

    use burn_tensor::{backend::Backend, Distribution, Shape, Tensor};
    use serial_test::serial;

    use crate::{kernel::prng::base::tests::calculate_bin_stats, tests::TestBackend, WgpuDevice};

    #[test]
    #[serial]
    fn subsequent_calls_give_different_tensors() {
        TestBackend::seed(0);
        let shape: Shape<2> = [40, 40].into();
        let device = WgpuDevice::default();

        let tensor_1 = Tensor::<TestBackend, 2>::random_device(
            shape.clone(),
            Distribution::Bernoulli(0.5),
            &device,
        );
        let tensor_2 = Tensor::<TestBackend, 2>::random_device(
            shape.clone(),
            Distribution::Bernoulli(0.5),
            &device,
        );
        let mut diff_exists = false;
        for i in 0..shape.num_elements() {
            if tensor_1.to_data().value[i] != tensor_2.to_data().value[i] {
                diff_exists = true;
                break;
            }
        }
        assert!(diff_exists);
    }

    #[test]
    #[serial]
    fn number_of_1_proportional_to_prob() {
        TestBackend::seed(0);
        let shape: Shape<2> = [40, 40].into();
        let device = WgpuDevice::default();
        let prob = 0.7;

        let tensor_1 = Tensor::<TestBackend, 2>::random_device(
            shape.clone(),
            Distribution::Bernoulli(prob),
            &device,
        );

        // High bound slightly over 1 so 1.0 is included in second bin
        let bin_stats = calculate_bin_stats(tensor_1.into_data().value, 2, 0., 1.1);
        assert!(
            f32::abs((bin_stats[1].count as f32 / shape.num_elements() as f32) - prob as f32)
                < 0.05
        );
    }

    #[test]
    #[serial]
    fn runs_test() {
        TestBackend::seed(0);
        let shape = Shape::new([512, 512]);
        let device = WgpuDevice::default();
        let tensor =
            Tensor::<TestBackend, 2>::random_device(shape, Distribution::Bernoulli(0.5), &device);

        let numbers = tensor.into_data().value;
        let stats = calculate_bin_stats(numbers, 2, 0., 1.1);
        let n_0 = stats[0].count as f32;
        let n_1 = stats[1].count as f32;
        let n_runs = (stats[0].n_runs + stats[1].n_runs) as f32;

        let expectation = (2. * n_0 * n_1) / (n_0 + n_1) + 1.0;
        let variance = ((2. * n_0 * n_1) * (2. * n_0 * n_1 - n_0 - n_1))
            / ((n_0 + n_1).powf(2.) * (n_0 + n_1 - 1.));
        let z = (n_runs - expectation) / variance.sqrt();

        // below 2 means we can have good confidence in the randomness
        // we put 2.5 to make sure it passes even when very unlucky
        assert!(z.abs() < 2.5);
    }
}
