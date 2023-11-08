use super::{build_info, KernelSettings, SourceTemplate, StaticKernelSource, WORKGROUP_DEFAULT};
use crate::{
    compute::StaticKernel, element::WgpuElement, kernel::elemwise_workgroup, kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(ReductionDimRaw, "../template/reduction/reduce_dim_alt.wgsl");

pub struct SumDim;

impl StaticKernelSource for SumDim {
    fn source() -> SourceTemplate {
        ReductionDimRaw::source().register(
            "shared_size",
            (WORKGROUP_DEFAULT * WORKGROUP_DEFAULT).to_string(),
        )
    }
}

/// Execute the sum dim kernel, new implementation
pub fn sum_dim_alt<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    reduction_dim::<SumDim, E, D>(input, dim)
}

fn reduction_dim<K: StaticKernelSource, E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let num_elems = shape_out.num_elements();
    let handle = input.client.empty(num_elems * core::mem::size_of::<E>());
    let output = WgpuTensor::new(
        input.client.clone(),
        input.device.clone(),
        shape_out.clone(),
        handle,
    );

    // on a [2000, 2000, 2000]
    let sum_group_size = input.shape.dims[dim]; // 2000
    let n_sum_groups = shape_out.num_elements(); // 4000000
    let num_invocation_per_workgroup = WORKGROUP_DEFAULT * WORKGROUP_DEFAULT; // 1024
                                                                              // ceil(2000/1024) = 2
    let n_input_values_per_thread =
        f32::ceil(sum_group_size as f32 / num_invocation_per_workgroup as f32) as u32;

    // if sum_group_size < num_elem_per_invocation, some threads will be idle
    // also, some threads can be idle on their last input values if it does not divide evenly

    // we want n_sum_groups workgroups
    // the grid has n_sum_groups workgroups
    // a workgroup has num_invocation_per_workgroup invocations/threads
    let grid = elemwise_workgroup(n_sum_groups, WORKGROUP_DEFAULT);

    // optimization to do after: have workgroups that do several sum_groups, leveraging idleness well
    // because 4000000 is too many workgroups!

    let kernel =
        StaticKernel::<KernelSettings<K, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>>::new(
            grid,
        );

    let mut info = build_info(&[&input, &output]);
    info.push(dim as u32);
    info.push(n_input_values_per_thread);
    info.push(2);
    let info_handle = input.client.create(bytemuck::cast_slice(&info));

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
    fn reduction_sum_dim_alt_simplest() {
        let tensor = Tensor::<TestBackend, 1>::random([700], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 1>::from_data(tensor.to_data());

        let val = Tensor::<TestBackend, 1>::from_primitive(reduction_dim::<SumDim, f32, 1>(
            tensor.into_primitive(),
            0,
        ));
        let val_ref = tensor_ref.sum_dim(0);
        println!("{:?}", val_ref);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
        // assert!(false);
    }

    #[test]
    fn reduction_sum_dim_alt_mid() {
        let tensor = Tensor::<TestBackend, 2>::random([6, 1024], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data());

        let val = Tensor::<TestBackend, 2>::from_primitive(reduction_dim::<SumDim, f32, 2>(
            tensor.into_primitive(),
            1,
        ));
        let val_ref = tensor_ref.sum_dim(1);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }

    #[test]
    fn reduction_sum_dim_alt_large() {
        let tensor = Tensor::<TestBackend, 3>::random([50, 1024, 50], Distribution::Default);
        let tensor_ref = Tensor::<ReferenceBackend, 3>::from_data(tensor.to_data());

        let val = Tensor::<TestBackend, 3>::from_primitive(reduction_dim::<SumDim, f32, 3>(
            tensor.into_primitive(),
            1,
        ));
        let val_ref = tensor_ref.sum_dim(1);

        val_ref.into_data().assert_approx_eq(&val.into_data(), 3);
    }
}
