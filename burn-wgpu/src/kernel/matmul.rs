use super::{build_info, KernelSettings};
use crate::{context::WorkGroup, element::WGPUElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;
use std::sync::Arc;

kernel_wgsl!(MatmulRaw, "../template/matmul.wgsl");

pub fn matmul<E: WGPUElement, const D: usize>(
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
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, Arc::new(buffer));
    let kernel = lhs
        .context
        .compile::<KernelSettings<MatmulRaw, E, i32, 1, 16, 16>>();

    let info = build_info(&[&lhs, &rhs]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup = WorkGroup::new(
        num_iter as u32,
        f32::ceil((lhs.shape.dims[D - 2] as f32) / 16.) as u32,
        f32::ceil((rhs.shape.dims[D - 1] as f32) / 16.) as u32,
    );

    lhs.context.execute(
        &workgroup,
        &kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}
