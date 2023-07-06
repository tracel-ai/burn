use crate::{
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(MaskFill, "../../template/mask/fill.wgsl");
kernel_wgsl!(MaskFillInplace, "../../template/mask/fill_inplace.wgsl");

pub fn mask_fill<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: E,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let num_elems = input.shape.num_elements();
    let buffer = input
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), input.shape.clone(), buffer);

    let value_buffer = input.context.create_buffer_with_data(E::as_bytes(&[value]));
    let kernel = input
        .context
        .compile_static::<KernelSettings<MaskFill, E, i32, WORKGROUP, WORKGROUP, 1>>();
    let mask = WgpuTensor::new(mask.context, mask.shape, mask.buffer);
    let info = build_info(&[&input, &mask, &output]);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[
            &input.buffer,
            &value_buffer,
            &mask.buffer,
            &output.buffer,
            &info_buffers,
        ],
    );

    output
}

pub fn mask_fill_inplace<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: E,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let num_elems = input.shape.num_elements();
    let value_buffer = input.context.create_buffer_with_data(E::as_bytes(&[value]));
    let kernel = input
        .context
        .compile_static::<KernelSettings<MaskFillInplace, E, i32, WORKGROUP, WORKGROUP, 1>>();
    let mask = WgpuTensor::new(mask.context, mask.shape, mask.buffer);
    let info = build_info(&[&input, &mask]);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&input.buffer, &value_buffer, &mask.buffer, &info_buffers],
    );

    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Bool, Distribution, Tensor};

    #[test]
    fn mask_fill_should_work_with_multiple_invocations() {
        let (tensor, mask, tensor_ref, mask_ref) = inputs_mask_fill();

        let actual = Tensor::<TestBackend, 3>::from_primitive(mask_fill::<f32, 3>(
            tensor.into_primitive(),
            mask.into_primitive(),
            4.0,
        ));
        let expected = tensor_ref.mask_fill(mask_ref, 4.0);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    #[test]
    fn mask_fill_inplace_should_work_with_multiple_invocations() {
        let (tensor, mask, tensor_ref, mask_ref) = inputs_mask_fill();

        let actual = Tensor::<TestBackend, 3>::from_primitive(mask_fill_inplace::<f32, 3>(
            tensor.into_primitive(),
            mask.into_primitive(),
            4.0,
        ));
        let expected = tensor_ref.mask_fill(mask_ref, 4.0);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    #[allow(clippy::type_complexity)]
    fn inputs_mask_fill() -> (
        Tensor<TestBackend, 3>,
        Tensor<TestBackend, 3, Bool>,
        Tensor<ReferenceBackend, 3>,
        Tensor<ReferenceBackend, 3, Bool>,
    ) {
        let tensor = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Default);
        let mask = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Uniform(0., 1.))
            .lower_equal_elem(0.5);
        let tensor_ref = Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data());
        let mask_ref = Tensor::<ReferenceBackend, 3, Bool>::from_data(mask.to_data());

        (tensor, mask, tensor_ref, mask_ref)
    }
}
