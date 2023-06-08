use super::{KernelGenerator, KernelSettings};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;
use std::sync::Arc;

kernel_wgsl!(GlobalReductionRaw, "../template/reduction/global.wgsl");
kernel_wgsl!(SumRaw, "../template/reduction/sum.wgsl");

struct SumGlobalReduction;

impl KernelGenerator for SumGlobalReduction {
    type Source = String;

    fn generate() -> Self::Source {
        GlobalReductionRaw::generate().replace("BODY", "sum_reduction(workgroup_id.x);")
            + SumRaw::generate().as_ref()
    }
}

pub fn reduction_sum<E: WgpuElement, const D: usize>(input: WgpuTensor<E, D>) -> WgpuTensor<E, 1> {
    reduction::<E, D, SumGlobalReduction>(input)
}

/// Perform a binary reduction by lauching multiple compute shaders reducing the input tensors
/// until the size becomes 1.
fn reduction<E: WgpuElement, const D: usize, K: KernelGenerator>(
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
