use std::{
    cmp::{max, min},
    sync::Arc,
};

use burn_tensor::Shape;
use wgpu::ComputePipeline;

use crate::{
    context::{Context, WorkGroup},
    element::WgpuElement,
    kernel::{build_info, SourceTemplate},
    tensor::WgpuTensor,
};

use super::padding::{crop, pad_round};

const MAX_SHARED_MEMORY_SIZE: usize = 8192;

pub(super) fn empty_from_context<E: WgpuElement, const D: usize>(
    context: Arc<Context>,
    shape: &Shape<D>,
) -> WgpuTensor<E, D> {
    let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<E>());

    WgpuTensor::new(context, shape.clone(), buffer)
}

pub(super) fn register_template<
    const B_M: usize,
    const B_N: usize,
    const B_K: usize,
    const T_M: usize,
    const T_N: usize,
    const WORKGROUP_SIZE_X: usize,
    const WORKGROUP_SIZE_Y: usize,
>(
    template: SourceTemplate,
) -> SourceTemplate {
    template
        .register("b_m", B_M.to_string())
        .register("b_n", B_N.to_string())
        .register("b_k", B_K.to_string())
        .register("bm_x_bk", (B_M * B_K).to_string())
        .register("bk_x_bn", (B_K * B_N).to_string())
        .register("t_m", T_M.to_string())
        .register("t_n", T_N.to_string())
        .register("tm_x_tn", (T_M * T_N).to_string())
}

pub(super) fn matmul_parameter_assertions<E: WgpuElement, const D: usize>(
    b_m: usize,
    b_n: usize,
    b_k: usize,
    t_m: usize,
    t_n: usize,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    lhs: &WgpuTensor<E, D>,
    rhs: &WgpuTensor<E, D>,
) {
    assert!(b_k <= min(b_m, b_n), "B_K must be smaller than both B_M and B_M, otherwise there won't be enough threads to fill shared memory. ");
    assert!(b_k * max(b_m, b_n) <= MAX_SHARED_MEMORY_SIZE, "B_K x B_M and B_K x B_N must be smaller or equal than 8192, otherwise shared memory limit will be busted. ");
    assert!(
        b_m % t_m == 0 && b_n % t_n == 0,
        "T_M must divide B_M in this version"
    );
    assert!(
        workgroup_size_x == b_m / t_m,
        "Workgroup size x must equal B_M / T_M"
    );
    assert!(
        workgroup_size_y == b_n / t_n,
        "Workgroup size y must equal B_N / T_N"
    );
    lhs.assert_is_on_same_device(&rhs);
}

pub(super) fn shape_out<E: WgpuElement, const D: usize>(
    lhs: &WgpuTensor<E, D>,
    rhs: &WgpuTensor<E, D>,
) -> Shape<D> {
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
    Shape::new(shape_out)
}

pub(super) fn make_workgroup<const D: usize>(
    output_shape: Shape<D>,
    b_m: usize,
    b_n: usize,
) -> WorkGroup {
    let num_blocks_x = f32::ceil(output_shape.dims[D - 2] as f32 / b_m as f32) as u32;
    let num_blocks_y = f32::ceil(output_shape.dims[D - 1] as f32 / b_n as f32) as u32;
    let mut num_blocks_z = 1;
    for i in 0..D - 2 {
        num_blocks_z *= output_shape.dims[i];
    }

    WorkGroup::new(num_blocks_x, num_blocks_y, num_blocks_z as u32)
}

pub(super) fn make_info_buffers<E: WgpuElement, const D: usize>(
    lhs: &WgpuTensor<E, D>,
    rhs: &WgpuTensor<E, D>,
    output: &WgpuTensor<E, D>,
) -> Arc<wgpu::Buffer> {
    let info = build_info(&[&lhs, &rhs, &output]);
    rhs.context
        .create_buffer_with_data(bytemuck::cast_slice(&info))
}

pub(super) fn matmul_tiling_2d_launch<
    E: WgpuElement,
    const D: usize,
    const B_M: usize,
    const B_N: usize,
    const B_K: usize,
    const T_M: usize,
    const T_N: usize,
    const WORKGROUP_SIZE_X: usize,
    const WORKGROUP_SIZE_Y: usize,
>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
    kernel: Arc<ComputePipeline>,
) -> WgpuTensor<E, D> {
    matmul_parameter_assertions::<E, D>(
        B_M,
        B_N,
        B_K,
        T_M,
        T_N,
        WORKGROUP_SIZE_X,
        WORKGROUP_SIZE_Y,
        &lhs,
        &rhs,
    );

    let final_output_shape = shape_out(&lhs, &rhs);
    let lhs = pad_round(lhs, B_M, B_K);
    let rhs = pad_round(rhs, B_K, B_N);
    let rounded_output_shape = shape_out(&lhs, &rhs);

    let output = empty_from_context::<E, D>(rhs.context.clone(), &rounded_output_shape);

    let workgroup = make_workgroup(rounded_output_shape, B_M, B_N);
    let info_buffers = make_info_buffers(&lhs, &rhs, &output);

    lhs.context.execute(
        workgroup,
        kernel,
        &[&lhs.buffer, &rhs.buffer, &output.buffer, &info_buffers],
    );

    crop(output, final_output_shape)
}
