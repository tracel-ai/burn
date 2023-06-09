use super::{build_info, DynamicKernelSettings, StaticKernelGenerator};
use crate::{context::WorkGroup, element::WgpuElement, kernel_wgsl, tensor::WgpuTensor};
use burn_tensor::Shape;
use std::sync::Arc;

const TILE_SIZE: usize = 32;

kernel_wgsl!(MatmulTiledRaw, "../template/matmul_tiled_2.wgsl");

fn workgroup_size_max(workgroup_size_x: usize, workgroup_size_y: usize) -> (usize, usize, usize) {
    let num_invocations = workgroup_size_x * workgroup_size_y;

    let factor = f32::ceil(num_invocations as f32 / 1024 as f32) as usize;
    if factor > 1 {
        return (workgroup_size_x / factor, workgroup_size_y / factor, factor);
    }

    (workgroup_size_x, workgroup_size_y, factor)
}

struct MatmulTiled;

impl StaticKernelGenerator for MatmulTiled {
    type Source = String;

    fn generate() -> Self::Source {
        MatmulTiledRaw::generate().replace("TILE_SIZE", &TILE_SIZE.to_string())
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
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, Arc::new(buffer));
    let num_rows = lhs.shape.dims[D - 2];
    let num_cols = rhs.shape.dims[D - 1];

    let workgroup_size_x = f32::ceil(num_rows as f32 / TILE_SIZE as f32) as usize;
    let workgroup_size_y = f32::ceil(num_cols as f32 / TILE_SIZE as f32) as usize;

    let (workgroup_size_x, workgroup_size_y, factor) =
        workgroup_size_max(workgroup_size_x, workgroup_size_y);

    let kernel = DynamicKernelSettings::<MatmulTiled, E, i32>::new(TILE_SIZE, TILE_SIZE, 1);

    let kernel = lhs.context.compile_dynamic(kernel);

    let info = build_info(&[&lhs, &rhs]);
    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup = WorkGroup::new(
        (workgroup_size_x * workgroup_size_y * factor) as u32,
        1,
        num_iter as u32,
    );

    // println!(
    //     "Workgroup {:?} - {} {}",
    //     workgroup, workgroup_size_x, workgroup_size_y
    // );
    // for x in 0..(TILE_SIZE * TILE_SIZE * factor) {
    //     let row = x / TILE_SIZE;
    //     let col = x % TILE_SIZE;
    //     if row >= num_rows || col >= num_cols {
    //         continue;
    //     }
    //     println!("{x} => {row}-{col}");
    // }

    lhs.context.execute(
        &workgroup,
        &kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}
