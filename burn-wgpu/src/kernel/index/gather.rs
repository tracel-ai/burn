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
    let indices = kernel::into_continuous(indices);

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
    use burn_tensor::{backend::Backend, Distribution, Int, Tensor};

    #[test]
    fn gather_should_work_with_multiple_workgroups() {
        TestBackend::seed(0);
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Standard);
        let indices = Tensor::<TestBackend, 1, Int>::from_data(
            Tensor::<TestBackend, 1>::random([6 * 256], Distribution::Uniform(0., 256.))
                .into_data()
                .convert(),
        )
        .reshape([6, 256]);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());
        let indices_ref =
            Tensor::<ReferenceBackend, 2, Int>::from_data(indices.to_data().convert());

        let actual = Tensor::<TestBackend, 2>::from_primitive(gather(
            1,
            tensor.into_primitive(),
            indices.into_primitive(),
        ));
        let expected = tensor_ref.gather(1, indices_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
