use burn_tensor::Shape;

use crate::{
    context::WorkGroup,
    element::WgpuElement,
    kernel::{prng::base::get_seeds, KernelSettings},
    kernel_wgsl,
    pool::get_context,
    tensor::WgpuTensor,
    GraphicsApi, WgpuDevice,
};

kernel_wgsl!(NormalPRNG, "../../template/prng/normal.wgsl");

/// Pseudo-random generator for normal distribution
pub fn random_normal<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
    mean: E,
    std: E,
) -> WgpuTensor<E, D> {
    let context = get_context::<G>(device);
    const WORKGROUP: usize = 32;
    const N_VALUES_PER_THREAD: u32 = 128; // must be even

    let num_elems = shape.num_elements();
    let num_threads = f32::ceil(num_elems as f32 / N_VALUES_PER_THREAD as f32);
    let num_invocations = f32::ceil(num_threads / (WORKGROUP * WORKGROUP) as f32);
    let workgroup_x = f32::ceil(f32::sqrt(num_invocations));
    let workgroup_y = f32::ceil(num_invocations / workgroup_x);
    let workgroup = WorkGroup::new(workgroup_x as u32, workgroup_y as u32, 1);

    let buffer = context.create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(context.clone(), shape, buffer);

    let mut info = get_seeds();
    info.insert(0, N_VALUES_PER_THREAD);
    let info_buffer = context.create_buffer_with_data(bytemuck::cast_slice(&info));

    let args = [mean, std];
    let args_buffer = context.create_buffer_with_data(E::as_bytes(&args));

    let kernel =
        context.compile_static::<KernelSettings<NormalPRNG, E, i32, WORKGROUP, WORKGROUP, 1>>();

    context.execute(
        workgroup,
        kernel,
        &[&output.buffer, &info_buffer, &args_buffer],
    );

    output
}

#[cfg(test)]
mod tests {

    use burn_tensor::{backend::Backend, Data, Distribution, Shape, Tensor};

    use crate::{kernel::prng::base::tests::calculate_bin_stats, tests::TestBackend, WgpuDevice};

    #[test]
    fn subsequent_calls_give_different_tensors() {
        TestBackend::seed(0);
        let shape = [4, 5];
        let device = WgpuDevice::default();

        let tensor_1 =
            Tensor::<TestBackend, 2>::random_device(shape, Distribution::Normal(0., 1.), &device);
        let tensor_2 =
            Tensor::<TestBackend, 2>::random_device(shape, Distribution::Normal(0., 1.), &device);
        for i in 0..20 {
            assert!(tensor_1.to_data().value[i] != tensor_2.to_data().value[i]);
        }
    }

    #[test]
    fn empirical_mean_close_to_expectation() {
        TestBackend::seed(0);
        let shape = [128, 128];
        let device = WgpuDevice::default();
        let mean = 10.;
        let tensor =
            Tensor::<TestBackend, 2>::random_device(shape, Distribution::Normal(mean, 2.), &device);
        let empirical_mean = tensor.mean().into_data();
        empirical_mean.assert_approx_eq(&Data::from([mean as f32]), 1);
    }

    #[test]
    fn normal_respects_68_95_99_rule() {
        // https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
        let shape: Shape<2> = [1000, 1000].into();
        let device = WgpuDevice::default();
        let mu = 0.;
        let s = 1.;
        let tensor = Tensor::<TestBackend, 2>::random_device(
            shape.clone(),
            Distribution::Normal(mu, s),
            &device,
        );
        let stats = calculate_bin_stats(
            tensor.into_data().value,
            6,
            (mu - 3. * s) as f32,
            (mu + 3. * s) as f32,
        );
        let assert_approx_eq = |count, percent| {
            let expected = percent * shape.num_elements() as f32 / 100.;
            assert!(f32::abs(count as f32 - expected) < 1000.);
        };
        assert_approx_eq(stats[0].count, 2.1);
        assert_approx_eq(stats[1].count, 13.6);
        assert_approx_eq(stats[2].count, 34.1);
        assert_approx_eq(stats[3].count, 34.1);
        assert_approx_eq(stats[4].count, 13.6);
        assert_approx_eq(stats[5].count, 2.1);
    }
}
