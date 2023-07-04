use crate::{
    element::WgpuElement,
    kernel::{self, build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(Scatter, "../../template/index/scatter.wgsl");

pub(crate) fn scatter<E: WgpuElement, I: WgpuElement, const D: usize>(
    dim: usize,
    tensor: WgpuTensor<E, D>,
    indexes: WgpuTensor<I, D>,
    value: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let indexes = kernel::into_continuous(indexes);
    let tensor = kernel::into_continuous(tensor);
    let value = kernel::into_continuous(value);
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };
    let mut info = build_info(&[&tensor]);
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
        .compile_static::<KernelSettings<Scatter, E, i32, WORKGROUP, WORKGROUP, 1>>();

    let workgroup = elemwise_workgroup(num_elems_per_workgroup, WORKGROUP);
    println!("{workgroup:?}");
    tensor.context.execute(
        workgroup,
        kernel,
        &[&tensor.buffer, &indexes.buffer, &value.buffer, &info_buffer],
    );

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{backend::Backend, Distribution, Int, Tensor};

    #[test]
    fn scatter_should_work_with_multiple_workgroups() {
        TestBackend::seed(0);
        let tensor = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Standard);
        let value = Tensor::<TestBackend, 2>::random([6, 256], Distribution::Standard);
        let indices = Tensor::<TestBackend, 1, Int>::from_data(
            Tensor::<TestBackend, 1>::random([6 * 256], Distribution::Uniform(0., 256.))
                .into_data()
                .convert(),
        )
        .reshape([6, 256]);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());
        let value_ref = Tensor::<ReferenceBackend, 2>::from_data(value.to_data());
        let indices_ref =
            Tensor::<ReferenceBackend, 2, Int>::from_data(indices.to_data().convert());

        let actual = Tensor::<TestBackend, 2>::from_primitive(scatter(
            1,
            tensor.into_primitive(),
            indices.into_primitive(),
            value.into_primitive(),
        ));
        let expected = tensor_ref.scatter(1, indices_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
