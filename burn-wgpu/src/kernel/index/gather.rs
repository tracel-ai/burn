use crate::{
    element::WgpuElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(Gather, "../../template/index/gather.wgsl");

pub(crate) fn gather<E: WgpuElement, I: WgpuElement, const D: usize>(
    dim: usize,
    tensor: WgpuTensor<E, D>,
    indices: WgpuTensor<I, D>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let shape_output = indices.shape.clone();
    let num_elems = shape_output.num_elements();
    let indices = kernel::into_contiguous(indices);

    let buffer = tensor
        .context
        .create_buffer(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(tensor.context.clone(), shape_output, buffer);
    let mut info = build_info(&[&tensor, &output]);
    info.push(dim as u32);
    let info_buffer = tensor
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let kernel = tensor
        .context
        .compile_static::<KernelSettings<Gather, E, i32, WORKGROUP, WORKGROUP, 1>>();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, Distribution, Int, Shape, Tensor};

    #[test]
    fn gather_should_work_with_multiple_workgroups_dim0() {
        test_same_as_ref([6, 256], 0);
    }

    #[test]
    fn gather_should_work_with_multiple_workgroups_dim1() {
        test_same_as_ref([6, 256], 1);
    }

    fn test_same_as_ref<const D: usize>(shape: [usize; D], dim: usize) {
        TestBackend::seed(0);
        let max = shape[dim];
        let shape = Shape::new(shape);
        let tensor = Tensor::<TestBackend, D>::random(shape.clone(), Distribution::Default);
        let indices = Tensor::<TestBackend, 1, Int>::from_data(
            Tensor::<TestBackend, 1>::random(
                [shape.num_elements()],
                Distribution::Uniform(0., max as f32),
            )
            .into_data()
            .convert(),
        )
        .reshape(shape);
        let tensor_ref = Tensor::<ReferenceBackend, D>::from_data(tensor.to_data());
        let indices_ref =
            Tensor::<ReferenceBackend, D, Int>::from_data(indices.to_data().convert());

        let actual = Tensor::<TestBackend, D>::from_primitive(gather(
            dim,
            tensor.into_primitive(),
            indices.into_primitive(),
        ));
        let expected = tensor_ref.gather(dim, indices_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
