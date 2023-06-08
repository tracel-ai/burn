use super::{build_info, KernelGenerator, KernelSettings};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;
use std::sync::Arc;

kernel_wgsl!(GlobalReductionRaw, "../template/reduction/global.wgsl");
kernel_wgsl!(SumRaw, "../template/reduction/sum.wgsl");
kernel_wgsl!(SumDimRaw, "../template/reduction/sum_dim.wgsl");
kernel_wgsl!(MeanDimRaw, "../template/reduction/mean_dim.wgsl");

struct SumGlobalReduction;

impl KernelGenerator for SumGlobalReduction {
    type Source = String;

    fn generate() -> Self::Source {
        GlobalReductionRaw::generate().replace("BODY", "sum_reduction(workgroup_id.x);")
            + SumRaw::generate().as_ref()
    }
}

pub fn reduction_sum<E: WgpuElement, const D: usize>(input: WgpuTensor<E, D>) -> WgpuTensor<E, 1> {
    reduction_recursive::<E, D, SumGlobalReduction>(input)
}

pub fn reduction_sum_dim<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let buffer = input
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), shape_out, Arc::new(buffer));

    let kernel = input
        .context
        .compile::<KernelSettings<SumDimRaw, E, i32, 256, 1, 1>>();
    let mut info = build_info(&[&input, &output]);
    info.push(dim as u32);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        &WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&input.buffer, &output.buffer, &info_buffers],
    );

    output
}

pub fn reduction_mean_dim<E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let buffer = input
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), shape_out, Arc::new(buffer));

    let kernel = input
        .context
        .compile::<KernelSettings<MeanDimRaw, E, i32, 256, 1, 1>>();
    let mut info = build_info(&[&input, &output]);
    info.push(dim as u32);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        &WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        &kernel,
        &[&input.buffer, &output.buffer, &info_buffers],
    );

    output
}

/// Perform a binary reduction by lauching multiple compute shaders reducing the input tensors
/// until the size becomes 1.
fn reduction_recursive<E: WgpuElement, const D: usize, K: KernelGenerator>(
    input: WgpuTensor<E, D>,
) -> WgpuTensor<E, 1> {
    const WORKGROUP: usize = 256;

    let mut input_buffer = input.buffer;
    let mut num_invocations =
        f32::ceil(input.shape.num_elements() as f32 / WORKGROUP as f32) as usize;

    let kernel = input
        .context
        .compile::<KernelSettings<K, E, i32, WORKGROUP, 1, 1>>();

    loop {
        let buffer = input
            .context
            .create_buffer(core::mem::size_of::<E>() * num_invocations);
        let workgroup = WorkGroup::new((num_invocations as usize) as u32, 1, 1);

        input
            .context
            .execute(&workgroup, &kernel, &[&input_buffer, &buffer]);

        if num_invocations == 1 {
            return WgpuTensor::new(input.context, Shape::new([1]), Arc::new(buffer));
        }

        input_buffer = Arc::new(buffer);
        num_invocations = f32::ceil(num_invocations as f32 / WORKGROUP as f32) as usize;
    }
}
