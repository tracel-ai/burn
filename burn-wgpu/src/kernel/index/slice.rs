use crate::{
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};
use burn_tensor::Shape;
use std::ops::Range;

kernel_wgsl!(IndexRaw, "../../template/index/slice.wgsl");
kernel_wgsl!(
    IndexAssignInplaceRaw,
    "../../template/index/slice_assign_inplace.wgsl"
);

pub(crate) fn slice<E: WgpuElement, const D1: usize, const D2: usize>(
    tensor: WgpuTensor<E, D1>,
    indices: [Range<usize>; D2],
) -> WgpuTensor<E, D1> {
    const WORKGROUP: usize = 32;

    let mut dims = tensor.shape.dims;
    for i in 0..D2 {
        dims[i] = indices[i].end - indices[i].start;
    }
    let shape_output = Shape::new(dims);
    let num_elems = shape_output.num_elements();

    let buffer = tensor
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(tensor.context.clone(), shape_output, buffer);
    let mut info = build_info(&[&tensor, &output]);

    for i in 0..D1 {
        let start = indices.get(i).map(|index| index.start).unwrap_or(0);
        info.push(start as u32);
    }

    let info_buffer = tensor
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = tensor
        .context
        .compile_static::<KernelSettings<IndexRaw, E, i32, WORKGROUP, WORKGROUP, 1>>();

    tensor.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&tensor.buffer, &output.buffer, &info_buffer],
    );

    output
}

pub(crate) fn slice_assign<E: WgpuElement, const D1: usize, const D2: usize>(
    tensor: WgpuTensor<E, D1>,
    indices: [Range<usize>; D2],
    value: WgpuTensor<E, D1>,
) -> WgpuTensor<E, D1> {
    const WORKGROUP: usize = 32;

    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };
    let num_elems = tensor.shape.num_elements();
    let mut info = build_info(&[&tensor, &value]);

    for i in 0..D1 {
        let start = indices.get(i).map(|index| index.start).unwrap_or(0);
        info.push(start as u32);
    }

    let info_buffer = tensor
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = tensor.context.compile_static::<KernelSettings<
        IndexAssignInplaceRaw,
        E,
        i32,
        WORKGROUP,
        WORKGROUP,
        1,
    >>();

    tensor.context.execute(
        elemwise_workgroup(num_elems, WORKGROUP),
        kernel,
        &[&tensor.buffer, &value.buffer, &info_buffer],
    );

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn slice_should_work_with_multiple_workgroups() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let indices = [3..5, 45..256];
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());

        let actual = slice(tensor.into_primitive(), indices.clone());
        let expected = tensor_ref.slice(indices);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn slice_assign_should_work_with_multiple_workgroups() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default);
        let value = Tensor::<TestBackend, 2>::random([2, 211], Distribution::Default);
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
