use crate::{
    compute::StaticKernel,
    element::JitElement,
    kernel::{build_info, elemwise_workgroup, KernelSettings},
    kernel_wgsl,
    tensor::JitTensor,
    Runtime,
};

use super::WORKGROUP_DEFAULT;

kernel_wgsl!(Cat, "../template/cat.wgsl");

pub fn cat<R: Runtime, E: JitElement, const D: usize>(
    inputs: Vec<JitTensor<R, E, D>>,
    dim: usize,
) -> JitTensor<R, E, D> {
    let first_input = inputs.first().unwrap();
    let client = &first_input.client;
    let mut shape_output = first_input.shape.clone();
    shape_output.dims[dim] = inputs.iter().map(|input| input.shape.dims[dim]).sum();

    let buffer = first_input
        .client
        .empty(shape_output.num_elements() * std::mem::size_of::<E>());

    let output = JitTensor::new(
        client.clone(),
        first_input.device.clone(),
        shape_output,
        buffer,
    );

    let mut dim_cat_index = 0;

    for input in inputs.iter() {
        let mut info = build_info(&[input, &output]);
        info.push(dim as u32);
        info.push(dim_cat_index as u32);
        dim_cat_index += input.shape.dims[dim];
        let info_buffer = client.create(bytemuck::cast_slice(&info));
        let kernel = StaticKernel::<
            KernelSettings<Cat, E, i32, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
        >::new(elemwise_workgroup(
            input.shape.num_elements(),
            WORKGROUP_DEFAULT,
        ));

        client.execute(
            Box::new(kernel),
            &[&input.handle, &output.handle, &info_buffer],
        );
    }

    output
}
