use crate::{
    codegen::{Elem, Operator, Variable},
    compute::StaticKernel,
    element::WgpuElement,
    kernel::WORKGROUP_DEFAULT,
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::WgpuTensor,
    unary,
};

use super::{elemwise_workgroup, KernelSettings};

kernel_wgsl!(Clamp, "../template/clamp/clamp.wgsl");
kernel_wgsl!(ClampInplace, "../template/clamp/clamp_inplace.wgsl");

// pub(crate) fn clamp_min<E: WgpuElement, const D: usize>(
//     input: WgpuTensor<E, D>,
//     min_value: E,
// ) -> WgpuTensor<E, D> {
//     todo!()
//     // unary!(
//     //     operator: |elem: Elem| Operator::ClampMin {
//     //         input: Variable::Input(0, elem),
//     //         out: Variable::Local(0, elem),
//     //     },
//     //     input: input; min_value,
//     //     elem: E
//     // )
// }
//
// pub(crate) fn clamp_max<E: WgpuElement, const D: usize>(
//     input: WgpuTensor<E, D>,
//     max_value: E,
// ) -> WgpuTensor<E, D> {
//     todo!()
// }

pub(crate) fn clamp<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    min_value: E,
    max_value: E,
) -> WgpuTensor<E, D> {
    let num_elems = input.shape.num_elements();
    let min_handle = input.client.create(E::as_bytes(&[min_value]));
    let max_handle = input.client.create(E::as_bytes(&[max_value]));

    if input.can_mut() {
        let kernel = StaticKernel::<
            KernelSettings<ClampInplace, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
        >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));

        input
            .client
            .execute(Box::new(kernel), &[&input.handle, &min_handle, &max_handle]);

        return input;
    }

    let output = empty_device(input.client.clone(), input.device.clone(), input.shape);
    let kernel = StaticKernel::<
        KernelSettings<Clamp, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(num_elems, WORKGROUP_DEFAULT));

    input.client.execute(
        Box::new(kernel),
        &[&input.handle, &output.handle, &min_handle, &max_handle],
    );

    output
}

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

    #[test]
    fn clamp_should_match_reference() {
        let input = Tensor::<TestBackend, 4>::random([1, 5, 32, 32], Distribution::Default);
        let input_ref = Tensor::<ReferenceBackend, 4>::from_data(input.to_data());

        let output = input.clamp(0.3, 0.7);

        output
            .into_data()
            .assert_approx_eq(&input_ref.clamp(0.3, 0.7).into_data(), 3);
    }
}
