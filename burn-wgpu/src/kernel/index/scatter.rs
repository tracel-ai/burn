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
    indices: WgpuTensor<I, D>,
    value: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 32;

    let indices = kernel::into_continuous(indices);
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
    fn scatter_should_work_with_multiple_workgroups_2d_dim0() {
        same_as_reference(0, [256, 32]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_2d_dim1() {
        same_as_reference(1, [32, 256]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_3d_dim0() {
        same_as_reference(0, [256, 6, 6]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_3d_dim1() {
        same_as_reference(1, [6, 256, 6]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_3d_dim2() {
        same_as_reference(2, [6, 6, 256]);
    }

    fn same_as_reference<const D: usize>(dim: usize, shape: [usize; D]) {
        TestBackend::seed(0);
        let tensor = Tensor::<TestBackend, D>::random(shape, Distribution::Standard);
        let value = Tensor::<TestBackend, D>::random(shape, Distribution::Standard);
        let indices = Tensor::<TestBackend, 1, Int>::from_data(
            Tensor::<TestBackend, 1>::random(
                [shape.iter().product()],
                Distribution::Uniform(0., shape[dim] as f32),
            )
            .into_data()
            .convert(),
        )
        .reshape(shape);
        let tensor_ref = Tensor::<ReferenceBackend, D>::from_data(tensor.to_data());
        let value_ref = Tensor::<ReferenceBackend, D>::from_data(value.to_data());
        let indices_ref =
            Tensor::<ReferenceBackend, D, Int>::from_data(indices.to_data().convert());

        let actual = Tensor::<TestBackend, D>::from_primitive(scatter(
            dim,
            tensor.into_primitive(),
            indices.into_primitive(),
            value.into_primitive(),
        ));
        let expected = tensor_ref.scatter(dim, indices_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
