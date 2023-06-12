use super::{build_info, DynamicKernelSettings, StaticKernelGenerator};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;

const BLOCK_SIZE: usize = 16;

kernel_wgsl!(
    MatmulCoalescingRaw,
    "../template/matmul_mem_coalescing.wgsl"
);

struct MatmulCoalescing;

impl StaticKernelGenerator for MatmulCoalescing {
    type Source = String;

    fn generate() -> Self::Source {
        MatmulCoalescingRaw::generate().replace("BLOCK_SIZE", &BLOCK_SIZE.to_string())
    }
}

pub fn matmul<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    lhs.assert_is_on_save_device(&rhs);
    let mut shape_out = [0; D];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });

    shape_out[D - 2] = lhs.shape.dims[D - 2];
    shape_out[D - 1] = rhs.shape.dims[D - 1];
    let shape_out = Shape::new(shape_out);

    let buffer = lhs
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);
    let num_rows = lhs.shape.dims[D - 2];
    let num_cols = rhs.shape.dims[D - 1];

    let kernel = DynamicKernelSettings::<MatmulCoalescing, E, i32>::new(BLOCK_SIZE, BLOCK_SIZE, 1);
    let kernel = lhs.context.compile_dynamic(kernel);

    let info = build_info(&[&lhs, &rhs]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup_x = f32::ceil(num_rows as f32 / BLOCK_SIZE as f32) as u32;
    let workgroup_y = f32::ceil(num_cols as f32 / BLOCK_SIZE as f32) as u32;
    let workgroup = WorkGroup::new(workgroup_x, workgroup_y, num_iter as u32);

    lhs.context.execute(
        workgroup,
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}
