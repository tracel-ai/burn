use super::{build_info, KernelSettings, SourceTemplate, StaticKernel};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;

kernel_wgsl!(RecursiveSumRaw, "../template/reduction/recursive_sum.wgsl");
kernel_wgsl!(ReductionDimRaw, "../template/reduction/reduce_dim.wgsl");
kernel_wgsl!(ReductionArgsRaw, "../template/reduction/args.wgsl");

pub struct ArgsMax;
pub struct ArgsMin;
pub struct SumDim;
pub struct MeanDim;

impl StaticKernel for SumDim {
    fn source_template() -> SourceTemplate {
        ReductionDimRaw::source_template().register("assign", "output[global_id.x] = sum;")
    }
}

impl StaticKernel for MeanDim {
    fn source_template() -> SourceTemplate {
        ReductionDimRaw::source_template()
            .add_template(
                "fn mean_dim(sum: {{ elem }}, dim: u32) -> {{ elem }} { 
    return sum / {{ elem }}(dim);
}",
            )
            .register("assign", "output[global_id.x] = mean_dim(sum, shape_dim);")
    }
}

impl StaticKernel for ArgsMax {
    fn source_template() -> SourceTemplate {
        ReductionArgsRaw::source_template()
            .register("cmp", ">")
            .register("initial", (-32767).to_string())
    }
}

impl StaticKernel for ArgsMin {
    fn source_template() -> SourceTemplate {
        ReductionArgsRaw::source_template()
            .register("cmp", "<")
            .register("initial", 32767.to_string())
    }
}

pub fn reduction_sum<E: WgpuElement, const D: usize>(input: WgpuTensor<E, D>) -> WgpuTensor<E, 1> {
    const WORKGROUP: usize = 256;

    let mut input_buffer = input.buffer;
    let mut num_invocations =
        f32::ceil(input.shape.num_elements() as f32 / WORKGROUP as f32) as usize;

    let kernel = input
        .context
        .compile_static::<KernelSettings<RecursiveSumRaw, E, i32, WORKGROUP, 1, 1>>();

    loop {
        let buffer = input
            .context
            .create_buffer(core::mem::size_of::<E>() * num_invocations);
        let workgroup = WorkGroup::new(num_invocations as u32, 1, 1);

        input
            .context
            .execute(workgroup, kernel.clone(), &[&input_buffer, &buffer]);

        if num_invocations == 1 {
            return WgpuTensor::new(input.context, Shape::new([1]), buffer);
        }

        input_buffer = buffer;
        num_invocations = f32::ceil(num_invocations as f32 / WORKGROUP as f32) as usize;
    }
}

pub fn reduction_dim<K: StaticKernel, E: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<E, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let buffer = input
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(input.context.clone(), shape_out, buffer);

    let kernel = input
        .context
        .compile_static::<KernelSettings<K, E, i32, 256, 1, 1>>();
    let mut info = build_info(&[&input, &output]);
    info.push(dim as u32);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&input.buffer, &output.buffer, &info_buffers],
    );

    output
}

pub fn reduction_args_dim<K: StaticKernel, E: WgpuElement, I: WgpuElement, const D: usize>(
    input: WgpuTensor<E, D>,
    dim: usize,
) -> WgpuTensor<I, D> {
    let mut shape_out = input.shape.clone();
    shape_out.dims[dim] = 1;
    let buffer = input
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<I>());
    let output = WgpuTensor::new(input.context.clone(), shape_out, buffer);

    let kernel = input
        .context
        .compile_static::<KernelSettings<K, E, I, 256, 1, 1>>();
    let mut info = build_info(&[&input, &output]);
    info.push(dim as u32);
    let info_buffers = input
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    input.context.execute(
        WorkGroup::new(
            f32::ceil(output.shape.num_elements() as f32 / 256_f32) as u32,
            1,
            1,
        ),
        kernel,
        &[&input.buffer, &output.buffer, &info_buffers],
    );

    WgpuTensor::new(output.context, output.shape, output.buffer)
}
