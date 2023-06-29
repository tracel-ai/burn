use std::cmp::{max, min};

use super::{build_info, SourceTemplate, StaticKernel};
use crate::{
    context::WorkGroup, element::WgpuElement, kernel::KernelSettings, kernel_wgsl,
    tensor::WgpuTensor,
};
use burn_tensor::Shape;

// Suppose a matmul of m1 of size [M, K] with m2 of size [K, N]
// Block size along dim M
const B_M: usize = 128;
// Block size along dim N
const B_N: usize = 128;
// Block size along dim K
const B_K: usize = 8;
// Tiling size along dim M
const T_M: usize = 8;
// Tiling size along dim N
const T_N: usize = 8;

// WORKGROUP_SIZE_X = ceil(B_M / T_M)
const WORKGROUP_SIZE_X: usize = 16;
// WORKGROUP_SIZE_Y = ceil(B_N / T_N)
const WORKGROUP_SIZE_Y: usize = 16;

const MAX_SHARED_MEMORY_SIZE: usize = 8192;

kernel_wgsl!(MatmulTiling2DRaw, "../template/matmul_blocktiling_2d.wgsl");

struct MatmulTiling2D;

impl StaticKernel for MatmulTiling2D {
    fn source_template() -> SourceTemplate {
        MatmulTiling2DRaw::source_template()
            .register("b_m", B_M.to_string())
            .register("b_n", B_N.to_string())
            .register("b_k", B_K.to_string())
            .register("bm_x_bk", (B_M * B_K).to_string())
            .register("bk_x_bn", (B_K * B_N).to_string())
            .register("t_m", T_M.to_string())
            .register("t_n", T_N.to_string())
            .register("tm_x_tn", (T_M * T_N).to_string())
    }
}

pub fn matmul<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    matmul_tiling_2d(lhs, rhs)
}

pub fn matmul_tiling_2d<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D> {
    assert!(B_K <= min(B_M, B_N), "B_K must be smaller than both B_M and B_M, otherwise there won't be enough threads to fill shared memory. ");
    assert!(B_K * max(B_M, B_N) <= MAX_SHARED_MEMORY_SIZE, "B_K x B_M and B_K x B_N must be smaller or equal than 8192, otherwise shared memory limit will be busted. ");
    lhs.assert_is_on_same_device(&rhs);

    let mut shape_out = [0; D];
    lhs.shape
        .dims
        .iter()
        .zip(rhs.shape.dims.iter())
        .enumerate()
        .for_each(|(index, (dim_lhs, dim_rhs))| {
            shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
        });

    let num_rows = lhs.shape.dims[D - 2];
    let num_cols = rhs.shape.dims[D - 1];
    shape_out[D - 2] = num_rows;
    shape_out[D - 1] = num_cols;
    let shape_out = Shape::new(shape_out);

    let buffer = lhs
        .context
        .create_buffer(shape_out.num_elements() * core::mem::size_of::<E>());
    let output = WgpuTensor::new(lhs.context.clone(), shape_out, buffer);

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / (WORKGROUP_SIZE_X * T_M) as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / (WORKGROUP_SIZE_Y * T_N) as f32) as u32;

    let kernel = lhs.context.compile_static::<KernelSettings<
        MatmulTiling2D,
        E,
        i32,
        WORKGROUP_SIZE_X,
        WORKGROUP_SIZE_Y,
        1,
    >>();

    let info = build_info(&[&lhs, &rhs, &output]);

    let info_buffers = lhs
        .context
        .create_buffer_with_data(bytemuck::cast_slice(&info));

    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output.shape.dims[i];
    }

    let workgroup = WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32);

    lhs.context.execute(
        workgroup,
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    output
}
