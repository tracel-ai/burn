use crate::{
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(IndexSelect, "../../template/index/index_select.wgsl");
kernel_wgsl!(
    IndexSelectAssignInplace,
    "../../template/index/index_select_assign_inplace.wgsl"
);

pub(crate) fn select<E: WgpuElement, I: WgpuElement, const D: usize>(
    tensor: WgpuTensor<E, D>,
    dim: usize,
    indexes: WgpuTensor<I, 1>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let mut output_shape = tensor.shape.clone();
    output_shape.dims[dim] = indexes.shape.dims[0];
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
            &indexes.buffer,
            &output.buffer,
            &info_buffer,
        ],
    );

    output
}

pub(crate) fn select_assign<E: WgpuElement, I: WgpuElement, const D: usize, const D2: usize>(
    tensor: WgpuTensor<E, D>,
    dim: usize,
    indexes: WgpuTensor<I, 1>,
    values: WgpuTensor<E, D2>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let mut shape = tensor.shape.clone();
    shape.dims[dim] = values.shape.dims[dim];
    let values = WgpuTensor::new(values.context, shape, values.buffer);
    let mut info = build_info(&[&tensor, &values]);
    info.push(dim as u32);

    let info_buffer = tensor
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = tensor.context.compile_static::<KernelSettings<
        IndexSelectAssignInplace,
        E,
        I,
        WORKGROUP,
        WORKGROUP,
        1,
    >>();

    let mut shape_tmp = values.shape;
    shape_tmp.dims[dim] = 1; // Just one thread for the dim.

    tensor.context.execute(
        elemwise_workgroup(shape_tmp.num_elements(), WORKGROUP),
        kernel,
        &[
            &tensor.buffer,
            &indexes.buffer,
            &values.buffer,
            &info_buffer,
        ],
    );

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Int, Tensor};

    #[test]
    fn slice_should_work_with_multiple_workgroups() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Standard);
        let indices = Tensor::<TestBackend, 1, Int>::arange(0..100);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());
        let indices_ref =
            Tensor::<ReferenceBackend, 1, Int>::from_data(indices.to_data().convert());

        let actual = select(tensor.into_primitive(), 1, indices.into_primitive());
        let expected = tensor_ref.select(1, indices);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn slice_assign_should_work_with_multiple_workgroups() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Standard);
        let value = Tensor::<TestBackend, 2>::random([2, 211], Distribution::Standard);
        let indices = [3..5, 45..256];
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());
        let value_ref = Tensor::<ReferenceBackend, 2>::from_data(value.to_data());

        let actual = slice_assign(
            tensor.into_primitive(),
            indices.clone(),
            value.into_primitive(),
        );
        let expected = tensor_ref.slice_assign(indices, value_ref);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }
}
