use burn_tensor::Shape;

use crate::{
    element::WgpuElement,
    kernel::{
        prng::base::{make_args_buffer, make_info_buffer, make_output_tensor},
        prng_workgroup, KernelSettings, SourceTemplate, StaticKernel,
    },
    pool::get_context,
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice,
};

use super::base::Prng;

struct UniformPrng;

impl StaticKernel for UniformPrng {
    fn source_template() -> SourceTemplate {
        Prng::source_template().register("num_args", "2").register(
            "prng_loop",
            include_str!("../../template/prng/uniform_inner_loop.wgsl"),
        )
    }
}

/// Pseudo-random generator for uniform distribution
pub fn random_uniform<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
    low: E,
    high: E,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;
    const N_VALUES_PER_THREAD: usize = 128;

    let context = get_context::<G>(device);
    let output = make_output_tensor(context.clone(), shape.clone());
    let info_buffer = make_info_buffer(context.clone(), N_VALUES_PER_THREAD);
    let args_buffer = make_args_buffer(context.clone(), &[low, high]);

    context.execute(
        prng_workgroup(shape.num_elements(), WORKGROUP, N_VALUES_PER_THREAD),
        context.compile_static::<KernelSettings<UniformPrng, E, i32, WORKGROUP, WORKGROUP, 1>>(),
        &[&output.buffer, &info_buffer, &args_buffer],
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
    #[serial]
    fn values_all_within_interval_default() {
        TestBackend::seed(0);
        let shape = [24, 24];
        let device = WgpuDevice::default();

        let tensor = Tensor::<TestBackend, 2>::random_device(shape, Distribution::Default, &device);
        tensor.to_data().assert_within_range(0..1);
    }

    #[test]
    #[serial]
    fn values_all_within_interval_uniform() {
        TestBackend::seed(0);
        let shape = [24, 24];
        let device = WgpuDevice::default();

        let tensor =
            Tensor::<TestBackend, 2>::random_device(shape, Distribution::Uniform(5., 17.), &device);
        tensor.to_data().assert_within_range(5..17);
    }

    #[test]
    #[serial]
    fn at_least_one_value_per_bin_uniform() {
        TestBackend::seed(0);
        let shape = [64, 64];
        let device = WgpuDevice::default();

        let tensor = Tensor::<TestBackend, 2>::random_device(
            shape,
            Distribution::Uniform(-5., 10.),
            &device,
        );
        let numbers = tensor.into_data().value;
        let stats = calculate_bin_stats(numbers, 3, -5., 10.);
        assert!(stats[0].count >= 1);
        assert!(stats[1].count >= 1);
        assert!(stats[2].count >= 1);
    }

    #[test]
    #[serial]
    fn runs_test() {
        TestBackend::seed(0);
        let shape = Shape::new([512, 512]);
        let device = WgpuDevice::default();
        let tensor = Tensor::<TestBackend, 2>::random_device(shape, Distribution::Default, &device);

        let numbers = tensor.into_data().value;
        let stats = calculate_bin_stats(numbers, 2, 0., 1.);
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
