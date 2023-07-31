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

kernel_wgsl!(UniformPRNG, "../../template/prng/uniform.wgsl");

/// Pseudo-random generator for default distribution (uniform in [0,1[)
pub fn random_uniform<G: GraphicsApi, E: WgpuElement, const D: usize>(
    shape: Shape<D>,
    device: &WgpuDevice,
    low: E,
    high: E,
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

    let args = [low, high];
    let args_buffer = context.create_buffer_with_data(E::as_bytes(&args));

    let kernel =
        context.compile_static::<KernelSettings<UniformPRNG, E, i32, WORKGROUP, WORKGROUP, 1>>();

    context.execute(
        workgroup,
        kernel,
        &[&output.buffer, &info_buffer, &args_buffer],
    );

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
    fn values_all_within_interval_default() {
        TestBackend::seed(0);
        let shape = [24, 24];
        let device = WgpuDevice::default();

        let tensor = Tensor::<TestBackend, 2>::random_device(shape, Distribution::Default, &device);
        tensor.to_data().assert_within_range(0..1);
    }

    #[test]
    fn values_all_within_interval_uniform() {
        TestBackend::seed(0);
        let shape = [24, 24];
        let device = WgpuDevice::default();

        let tensor =
            Tensor::<TestBackend, 2>::random_device(shape, Distribution::Uniform(5., 17.), &device);
        tensor.to_data().assert_within_range(5..17);
    }

    #[test]
    fn runs_test() {
        TestBackend::seed(0);
        let shape = Shape::new([512, 512]);
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

        // below 2 means we can have good confidence in the randomness
        assert!(z.abs() < 2.);
    }
}
