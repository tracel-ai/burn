use super::build_info;
use crate::{
    context::WorkGroup, element::WgpuElement, kernel::KernelSettings, kernel_wgsl,
    tensor::WgpuTensor,
};

kernel_wgsl!(MaskFill, "../template/mask/fill.wgsl");
kernel_wgsl!(MaskFillInplace, "../template/mask/fill_inplace.wgsl");
kernel_wgsl!(MaskWhere, "../template/mask/where.wgsl");
kernel_wgsl!(MaskWhereInplace, "../template/mask/where_inplace.wgsl");

pub fn mask_fill<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: E,
) -> WgpuTensor<E, D> {
    const WORKGROUP_SIZE: usize = 256;

    let buffer = input
        .context
        .create_buffer(input.shape.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), input.shape.clone(), buffer);

    let value_buffer = input.context.create_buffer_with_data(E::as_bytes(&[value]));
    let kernel = input
        .context
        .compile_static::<KernelSettings<MaskFill, E, i32, WORKGROUP_SIZE, 1, 1>>();
    let mask = WgpuTensor::new(mask.context, mask.shape, mask.buffer);
    let info = build_info(&[&input, &mask, &output]);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / WORKGROUP_SIZE as f32) as u32,
            1,
            1,
        ),
        kernel,
        &[
            &input.buffer,
            &value_buffer,
            &mask.buffer,
            &output.buffer,
            &info_buffers,
        ],
    );

    output
}

pub fn mask_fill_inplace<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: E,
) -> WgpuTensor<E, D> {
    const WORKGROUP_SIZE: usize = 256;

    let value_buffer = input.context.create_buffer_with_data(E::as_bytes(&[value]));
    let kernel = input
        .context
        .compile_static::<KernelSettings<MaskFillInplace, E, i32, WORKGROUP_SIZE, 1, 1>>();
    let mask = WgpuTensor::new(mask.context, mask.shape, mask.buffer);
    let info = build_info(&[&input, &mask]);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        WorkGroup::new(
            f32::ceil(input.shape.num_elements() as f32 / WORKGROUP_SIZE as f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&input.buffer, &value_buffer, &mask.buffer, &info_buffers],
    );

    input
}

pub fn mask_where<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    const WORKGROUP_SIZE: usize = 256;

    let buffer = input
        .context
        .create_buffer(input.shape.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), input.shape.clone(), buffer);

    let kernel = input
        .context
        .compile_static::<KernelSettings<MaskWhere, E, i32, WORKGROUP_SIZE, 1, 1>>();
    let mask = WgpuTensor::new(mask.context, mask.shape, mask.buffer);
    let info = build_info(&[&input, &value, &mask, &output]);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / WORKGROUP_SIZE as f32) as u32,
            1,
            1,
        ),
        kernel,
        &[
            &input.buffer,
            &value.buffer,
            &mask.buffer,
            &output.buffer,
            &info_buffers,
        ],
    );

    output
}

pub fn mask_where_inplace<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    mask: WgpuTensor<u32, D>,
    value: WgpuTensor<E, D>,
    direction: u32,
) -> WgpuTensor<E, D> {
    const WORKGROUP_SIZE: usize = 256;

    let kernel = input
        .context
        .compile_static::<KernelSettings<MaskWhereInplace, E, i32, WORKGROUP_SIZE, 1, 1>>();
    let mask = WgpuTensor::new(mask.context, mask.shape, mask.buffer);
    let mut info = build_info(&[&input, &value, &mask]);
    info.push(direction);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        WorkGroup::new(
            f32::ceil(input.shape.num_elements() as f32 / WORKGROUP_SIZE as f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&input.buffer, &value.buffer, &mask.buffer, &info_buffers],
    );

    input
}
