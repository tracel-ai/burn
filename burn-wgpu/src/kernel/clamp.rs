use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    tensor::WgpuTensor,
};

macro_rules! clamp_op {
    ($kernel_name:ident, $func_name:ident, $shader_path:expr) => {
        kernel_wgsl!($kernel_name, $shader_path);

        pub(crate) fn $func_name<E: WgpuElement, const D: usize>(
            input: WgpuTensor<E, D>,
            value: E,
        ) -> WgpuTensor<E, D> {
            let num_elems = input.shape.num_elements();
            let num_elems_buffer = input
                .client
                .create(bytemuck::cast_slice(&[num_elems as u32]));
            let value_handle = input.client.create(E::as_bytes(&[value]));

            let kernel = StaticKernel::<
                KernelSettings<$kernel_name, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
            >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));

            input.client.execute(
                Box::new(kernel),
                &[&input.handle, &value_handle, &num_elems_buffer],
            );
            input
        }
    };
}

clamp_op!(ClampMin, clamp_min, "../template/clamp/clamp_min.wgsl");
clamp_op!(ClampMax, clamp_max, "../template/clamp/clamp_max.wgsl");

#[cfg(test)]
mod tests {
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn clamp_min_should_match_reference() {
        let input = Tensor::<TestBackend, 4>::random([1, 5, 32, 32], Distribution::Default);
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());

        let output = input.clamp_min(0.5);

        output
            .into_data()
            .assert_approx_eq(&input_ref.clamp_min(0.5).into_data(), 3);
    }

    #[test]
    fn clamp_max_should_match_reference() {
        let input = Tensor::<TestBackend, 4>::random([1, 5, 32, 32], Distribution::Default);
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());

        let output = input.clamp_max(0.5);

        output
            .into_data()
            .assert_approx_eq(&input_ref.clamp_max(0.5).into_data(), 3);
    }
}
