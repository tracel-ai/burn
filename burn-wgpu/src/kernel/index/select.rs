use crate::{
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(IndexSelect, "../../template/index/select.wgsl");
kernel_wgsl!(
    SelectAssignInplace,
    "../../template/index/select_assign_inplace.wgsl"
);

pub(crate) fn select<E: WgpuElement, I: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    dim: usize,
    indices: WgpuTensor<I, 1>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let mut output_shape = tensor.shape.clone();
    output_shape.dims[dim] = indices.shape.dims[0];
    let num_elems = output_shape.num_elements();

    let buffer = tensor
        .context
        .create_buffer(num_elems * std::mem::size_of::<E>());
    let output = WgpuTensor::new(tensor.context.clone(), output_shape, buffer);

    let mut info = build_info(&[&tensor, &output]);
    info.push(dim as u32);

    let info_buffer = tensor
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = tensor
        .context
        .compile_static::<KernelSettings<IndexSelect, E, I, WORKGROUP, WORKGROUP, 1>>();

    tensor.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[
            &tensor.buffer,
            &indices.buffer,
            &output.buffer,
            &info_buffer,
        ],
    );

    output
}

pub(crate) fn select_assign<E: WgpuElement, I: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    dim: usize,
    indices: WgpuTensor<I, 1>,
    value: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let mut info = build_info(&[&tensor, &value]);
    let mut strides = [0; D];
    let mut current = 1;
    let mut num_elems_per_workgroup = 1;

    tensor
        .shape
        .dims
        .iter()
        .enumerate()
        .rev()
        .filter(|(index, _val)| *index != dim)
        .for_each(|(index, val)| {
            strides[index] = current;
            current *= val;
            num_elems_per_workgroup *= tensor.shape.dims[index];
        });

    strides
        .into_iter()
        .for_each(|stride| info.push(stride as u32));

    info.push(dim as u32);

    let info_buffer = tensor
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = tensor
        .context
        .compile_static::<KernelSettings<SelectAssignInplace, E, I, WORKGROUP, WORKGROUP, 1>>();

    tensor.context.execute(
        elemwise_workgroup(num_elems_per_workgroup, WORKGROUP),
        kernel,
        &[&tensor.buffer, &indices.buffer, &value.buffer, &info_buffer],
    );

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, Distribution, Int, Tensor};

    #[test]
    fn select_should_work_with_multiple_workgroups() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let indices = Tensor::<TestBackend, 1, Int>::arange(0..100);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());
        let indices_ref =
            Tensor::<ReferenceBackend, 1, Int>::from_data(indices.to_data().convert());

        let actual = select(tensor.into_primitive(), 1, indices.into_primitive());
        let expected = tensor_ref.select(1, indices_ref);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_2d_dim0() {
        select_assign_same_as_ref(0, [256, 6]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_2d_dim1() {
        select_assign_same_as_ref(1, [6, 256]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_3d_dim0() {
        select_assign_same_as_ref(0, [256, 6, 6]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_3d_dim1() {
        select_assign_same_as_ref(1, [6, 256, 6]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_3d_dim2() {
        select_assign_same_as_ref(2, [6, 6, 256]);
    }

    fn select_assign_same_as_ref<const D: usize>(dim: usize, shape: [usize; D]) {
        TestBackend::seed(0);
        let tensor = Tensor::<TestBackend, D>::random(shape, Distribution::Default);
        let value = Tensor::<TestBackend, D>::random(shape, Distribution::Default);
        let indices = Tensor::<TestBackend, 1, Int>::from_data(
            Tensor::<TestBackend, 1>::random(
                [shape[dim]],
                Distribution::Uniform(0., shape[dim] as f32),
            )
            .into_data()
            .convert(),
        );
        let tensor_ref = Tensor::<ReferenceBackend, D>::from_data(tensor.to_data());
        let value_ref = Tensor::<ReferenceBackend, D>::from_data(value.to_data());
        let indices_ref =
            Tensor::<ReferenceBackend, 1, Int>::from_data(indices.to_data().convert());

        let actual = Tensor::<TestBackend, D>::from_primitive(select_assign(
            tensor.into_primitive(),
            dim,
            indices.into_primitive(),
            value.into_primitive(),
        ));
        let expected = tensor_ref.select_assign(dim, indices_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
