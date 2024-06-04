use crate::{
    kernel::{into_contiguous, Kernel, SUBCUBE_DIM_APPROX},
    tensor::JitTensor,
    FloatElement, JitRuntime,
};
use burn_cube::ir::KernelDefinition;
use burn_cube::{frontend::TensorHandle, KernelSettings};

use super::simple_launch_options;
use burn_cube::prelude::*;

#[cube(launch)]
fn matmul_kernel<F: Float>(
    lhs: Tensor<F>,
    rhs: Tensor<F>,
    mut out: Tensor<F>,
    num_batches: Comptime<Option<UInt>>,
) {
    let rank = out.rank();
    let end = Comptime::unwrap_or_else(num_batches, || rank - UInt::new(2));
    let unroll = Comptime::is_some(num_batches);

    let n_rows = lhs.shape(rank - UInt::new(2));
    let n_cols = rhs.shape(rank - UInt::new(1));
    let k = rhs.shape(rank - UInt::new(2));

    let batch_pos = ABSOLUTE_POS_Z;
    let row = CUBE_DIM_X * CUBE_POS_X + UNIT_POS_X;
    let col = CUBE_DIM_Y * CUBE_POS_Y + UNIT_POS_Y;

    if row >= n_rows || col >= n_cols {
        return;
    }

    let vectorization_factor = Comptime::vectorization(out);

    let mut offset_lhs = UInt::new(0);
    let mut offset_rhs = UInt::new(0);
    let offset_out = n_rows * n_cols * batch_pos;

    for i in range(0u32, end, unroll) {
        let ogwl = offset_out * Comptime::runtime(vectorization_factor) / out.stride(i);

        offset_lhs += ogwl % lhs.shape(i) * lhs.stride(i);
        offset_rhs += ogwl % rhs.shape(i) * rhs.stride(i);
    }

    offset_lhs /= Comptime::runtime(vectorization_factor);
    offset_rhs /= Comptime::runtime(vectorization_factor);

    let mut sum = F::new(0.);

    for j in range(0u32, k, Comptime::new(false)) {
        let lhs_index = row * k + j + offset_lhs;
        let rhs_index = j * n_cols + col + offset_rhs;

        sum += lhs[lhs_index] * rhs[rhs_index];
    }

    let out_index = row * n_cols + col + offset_out;
    out[out_index] = sum;
}

/// Matrix multiplication using memory coalescing algorithm with workgroups of size 16
pub fn matmul_mem_coalescing_default<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    matmul_simple::<R, E, D>(lhs, rhs, out, SUBCUBE_DIM_APPROX, SUBCUBE_DIM_APPROX)
}

/// Matrix multiplication using memory coalescing algorithm with custom workgroup sizes
pub fn matmul_simple<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> JitTensor<R, E, D> {
    lhs.assert_is_on_same_device(&rhs);
    let lhs = into_contiguous(lhs);
    let rhs = into_contiguous(rhs);

    let workgroup = simple_launch_options(
        &lhs.shape,
        &rhs.shape,
        &out.shape,
        workgroup_size_x,
        workgroup_size_y,
    );

    let settings = KernelSettings::default()
        .vectorize_input(0, 4)
        .vectorize_output(0, 4);

    matmul_kernel_launch::<E::CubeElement, R>(
        lhs.client,
        workgroup,
        settings,
        TensorHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
        Some(UInt::new(D as u32 - 2)),
    );

    out
}
