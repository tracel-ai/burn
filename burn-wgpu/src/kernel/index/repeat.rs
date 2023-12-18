use crate::{
    compute::StaticKernel,
    element::WgpuElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings, WORKGROUP_DEFAULT},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(RepeatRaw, "../../template/index/repeat.wgsl");

pub(crate) fn repeat<E: WgpuElement, const D1: usize>(
    input: WgpuTensor<E, D1>,
    dim: usize,
    times: usize,
) -> WgpuTensor<E, D1> {
    let mut shape = input.shape.clone();
    if shape.dims[dim] != 1 {
        panic!("Can only repeat dimension with dim=1");
    }

    // Create output handle
    shape.dims[dim] = times;
    let num_elems_output = shape.num_elements();
    let handle = input
        .client
        .empty(num_elems_output * core::mem::size_of::<E>());
    let output = WgpuTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape.clone(),
        handle,
    );

    let mut info = build_info(&[&input, &output]);
    info.push(dim as u32);
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

    let kernel = StaticKernel::<
        KernelSettings<RepeatRaw, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(num_elems_output, WORKGROUP_DEFAULT));

    input.client.execute(
        Box::new(kernel),
        &[&input.handle, &output.handle, &info_handle],
    );

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend};
    use burn_tensor::{Distribution, Tensor};

    #[test]
    fn repeat_dim_0_few_times() {
        let tensor = Tensor::<TestBackend, 3>::random([1, 6, 6], Distribution::Default);
        let dim = 0;
        let times = 4;
        let tensor_ref = Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data());

        let actual = repeat(tensor.into_primitive(), dim, times);
        let expected = tensor_ref.repeat(dim, times);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 3>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn repeat_dim_1_few_times() {
        let tensor = Tensor::<TestBackend, 3>::random([6, 1, 6], Distribution::Default);
        let dim = 1;
        let times = 4;
        let tensor_ref = Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data());

        let actual = repeat(tensor.into_primitive(), dim, times);
        let expected = tensor_ref.repeat(dim, times);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 3>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn repeat_dim_2_few_times() {
        let tensor = Tensor::<TestBackend, 3>::random([6, 6, 1], Distribution::Default);
        let dim = 2;
        let times = 4;
        let tensor_ref = Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data());

        let actual = repeat(tensor.into_primitive(), dim, times);
        let expected = tensor_ref.repeat(dim, times);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 3>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn repeat_dim_2_many_times() {
        let tensor = Tensor::<TestBackend, 3>::random([10, 10, 1], Distribution::Default);
        let dim = 2;
        let times = 200;
        let tensor_ref = Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data());

        let actual = repeat(tensor.into_primitive(), dim, times);
        let expected = tensor_ref.repeat(dim, times);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 3>::from_primitive(actual).into_data(),
            3,
        );
    }
}
