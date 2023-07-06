use crate::{
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(MaskWhere, "../../template/mask/where.wgsl");
kernel_wgsl!(MaskWhereInplace, "../../template/mask/where_inplace.wgsl");

pub fn mask_where<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let num_elems = input.shape.num_elements();
    let buffer = input
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), input.shape.clone(), buffer);

    let kernel = input
        .context
        .compile_static::<KernelSettings<MaskWhere, E, i32, WORKGROUP, WORKGROUP, 1>>();
    let mask = WgpuTensor::new(mask.context, mask.shape, mask.buffer);
    let info = build_info(&[&input, &value, &mask, &output]);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[
            &input.buffer,
            &value.buffer,
            &mask.buffer,
            &output.buffer,
            &info_buffers,
        ],
    );

    output
}

pub fn mask_where_inplace<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: WgpuTensor<E, D>,
    reverse: bool,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let kernel = input
        .context
        .compile_static::<KernelSettings<MaskWhereInplace, E, i32, WORKGROUP, WORKGROUP, 1>>();
    let mask = WgpuTensor::new(mask.context, mask.shape, mask.buffer);
    let mut info = build_info(&[&input, &value, &mask]);
    info.push(match reverse {
        true => 1,
        false => 0,
    });
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        elemwise_workgroup(input.shape.num_elements(), WORKGROUP),
        kernel,
        &[&input.buffer, &value.buffer, &mask.buffer, &info_buffers],
    );

    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, Bool, Distribution, Tensor};

    #[test]
    fn mask_where_should_work_with_multiple_invocations() {
        let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

        let actual = Tensor::<TestBackend, 3>::from_primitive(mask_where::<f32, 3>(
            tensor.into_primitive(),
            mask.into_primitive(),
            value.into_primitive(),
        ));
        let expected = tensor_ref.mask_where(mask_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
    #[test]
    fn mask_where_inplace_direction_1_should_work_with_multiple_invocations() {
        let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

        let actual = Tensor::<TestBackend, 3>::from_primitive(mask_where_inplace::<f32, 3>(
            tensor.into_primitive(),
            mask.into_primitive(),
            value.into_primitive(),
            false,
        ));
        let expected = tensor_ref.mask_where(mask_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    #[test]
    fn mask_where_inplace_direction_0_should_work_with_multiple_invocation() {
        let (tensor, value, mask, tensor_ref, value_ref, mask_ref) = inputs_mask_where();

        let actual = Tensor::<TestBackend, 3>::from_primitive(mask_where_inplace::<f32, 3>(
            value.into_primitive(),
            mask.into_primitive(),
            tensor.into_primitive(),
            true,
        ));
        let expected = tensor_ref.mask_where(mask_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    #[allow(clippy::type_complexity)]
    fn inputs_mask_where() -> (
        Tensor<TestBackend, 3>,
        Tensor<TestBackend, 3>,
        Tensor<TestBackend, 3, Bool>,
        Tensor<ReferenceBackend, 3>,
        Tensor<ReferenceBackend, 3>,
        Tensor<ReferenceBackend, 3, Bool>,
    ) {
        TestBackend::seed(0);
        let tensor = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Default);
        let value = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Default);
        let mask = Tensor::<TestBackend, 3>::random([2, 6, 256], Distribution::Uniform(0., 1.))
            .lower_equal_elem(0.5);
        let tensor_ref = Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data());
        let value_ref = Tensor::<ReferenceBackend, 3>::from_data(value.to_data());
        let mask_ref = Tensor::<ReferenceBackend, 3, Bool>::from_data(mask.to_data());
        assert_eq!(mask.to_data(), mask_ref.to_data());

        (tensor, value, mask, tensor_ref, value_ref, mask_ref)
    }
}
