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

kernel_wgsl!(BernoulliPRNG, "../../template/prng/bernoulli.wgsl");

/// Pseudo-random generator for bernoulli
pub fn random_bernoulli<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
    prob: E,
) -> WgpuTensor<E, D> {
    let context = get_context::<G>(device);
    const WORKGROUP: usize = 32;
    const N_VALUES_PER_THREAD: u32 = 128;
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

    let args = [prob];
    let args_buffer = context.create_buffer_with_data(E::as_bytes(&args));

    let kernel =
        context.compile_static::<KernelSettings<BernoulliPRNG, E, i32, WORKGROUP, WORKGROUP, 1>>();

    context.execute(
        workgroup,
        kernel,
        &[&output.buffer, &info_buffer, &args_buffer],
    );

    output
}

#[cfg(test)]
mod tests {
    use core::f32;

    use burn_tensor::{backend::Backend, Distribution, Shape, Tensor};

    use crate::{kernel::prng::base::tests::calculate_bin_stats, tests::TestBackend, WgpuDevice};

    #[test]
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
                < 0.01
        );
    }

    #[test]
    fn runs_test() {
        TestBackend::seed(0);
        let shape = Shape::new([512, 512]);
        let device = WgpuDevice::default();
        let tensor = Tensor::<TestBackend, 2>::random_device(
            shape.clone(),
            Distribution::Bernoulli(0.5),
            &device,
        );

        let numbers = tensor.clone().into_data().value;
        let stats = calculate_bin_stats(numbers, 2, 0., 1.1);
        let n_0 = stats[0].count as f32;
        let n_1 = stats[1].count as f32;
        let n_runs = (stats[0].n_runs + stats[1].n_runs) as f32;

        let expectation = (2. * n_0 * n_1) / (n_0 + n_1) + 1.0;
        let variance = ((2. * n_0 * n_1) * (2. * n_0 * n_1 - n_0 - n_1))
            / ((n_0 + n_1).powf(2.) * (n_0 + n_1 - 1.));
        let z = (n_runs - expectation) / variance.sqrt();

        // below 2 means we can have good confidence in the randomness
        assert!(z.abs() < 2.);
    }
}
