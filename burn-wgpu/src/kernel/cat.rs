use crate::{
    context::WorkGroup,
    element::WgpuElement,
    kernel::{build_info, KernelSettings},
    kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(Cat, "../template/cat.wgsl");

pub fn cat<E: WgpuElement, const D: usize>(
    inputs: Vec<WgpuTensor<E, D>>,
    dim: usize,
) -> WgpuTensor<E, D> {
    const WORKGROUP: usize = 256;

    let first_input = inputs.get(0).unwrap();
    let context = &first_input.context;
    let mut shape_output = first_input.shape.clone();
    shape_output.dims[dim] = inputs.iter().map(|input| input.shape.dims[dim]).sum();

    let buffer = first_input
        .context
        .create_buffer(shape_output.num_elements() * std::mem::size_of::<E>());

    let output = WgpuTensor::new(context.clone(), shape_output, buffer);
    let kernel = context.compile_static::<KernelSettings<Cat, E, i32, WORKGROUP, 1, 1>>();

    let mut dim_cat_index = 0;

    for input in inputs.iter() {
        let workgroup = WorkGroup::new(
            f32::ceil(input.shape.num_elements() as f32 / WORKGROUP as f32) as u32,
            1,
            1,
        );
        let mut info = build_info(&[input, &output]);
        info.push(dim as u32);
        info.push(dim_cat_index as u32);
        dim_cat_index += input.shape.dims[dim];
        let info_buffer = context.create_buffer_with_data(bytemuck::cast_slice(&info));

        context.execute(
            workgroup,
            kernel.clone(),
            &[&input.buffer, &output.buffer, &info_buffer],
        );
    }

    output
}
