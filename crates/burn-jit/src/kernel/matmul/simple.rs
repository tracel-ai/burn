use crate::{
    kernel::{into_contiguous, Kernel, SUBCUBE_DIM_APPROX},
    ops::swap_dims,
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
    let mut k = rhs.shape(rank - UInt::new(2));

    let batch_pos = ABSOLUTE_POS_Z;
    let row = CUBE_DIM_X * CUBE_POS_X + UNIT_POS_X;
    let col = CUBE_DIM_Y * CUBE_POS_Y + UNIT_POS_Y;

    if row >= n_rows || col >= n_cols {
        return;
    }

    let vectorization_factor = Comptime::vectorization(lhs);

    let mut offset_lhs = UInt::new(0);
    let mut offset_rhs = UInt::new(0);
    let offset_out = n_rows * n_cols * batch_pos;

    for i in range(0u32, end, unroll) {
        let ogwl = offset_out / out.stride(i);

        offset_lhs += ogwl % lhs.shape(i) * lhs.stride(i);
        offset_rhs += ogwl % rhs.shape(i) * rhs.stride(i);
    }

    offset_lhs /= Comptime::runtime(vectorization_factor);
    offset_rhs /= Comptime::runtime(vectorization_factor);

    let mut sum = F::vectorized(0., Comptime::get(vectorization_factor));

    k /= Comptime::runtime(vectorization_factor);

    for i in range(0u32, k, Comptime::new(false)) {
        let lhs_index = row * k + i + offset_lhs;
        let rhs_index = col * k + i + offset_rhs;

        sum += lhs[lhs_index] * rhs[rhs_index];
    }

    let mut out_index = row * n_cols + col;
    out_index += offset_out;

    let unroll_sum = Comptime::map(vectorization_factor, |w: UInt| w != UInt::new(1));
    if Comptime::get(unroll_sum) {
        let mut accum = F::new(0.);
        for v in range(
            0u32,
            Comptime::get(vectorization_factor),
            Comptime::new(true),
        ) {
            accum += sum[v];
        }

        out[out_index] = accum;
    } else {
        out[out_index] = sum;
    }
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

    let rhs_original_shape = rhs.shape.clone();
    let rhs = into_contiguous(swap_dims(rhs, D - 1, D - 2));

    let workgroup = simple_launch_options(
        &lhs.shape,
        &rhs_original_shape,
        &out.shape,
        workgroup_size_x,
        workgroup_size_y,
    );

    let vectorization_factor = match lhs.shape.dims[D - 1] % 4 == 0 {
        true => 4,
        false => 1,
    };
    let settings = KernelSettings::default()
        .vectorize_input(0, vectorization_factor)
        .vectorize_input(1, vectorization_factor)
        .vectorize_output(0, 1);

    matmul_kernel_launch::<E::CubeElement, R>(
        lhs.client,
        workgroup,
        settings,
        TensorHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
        TensorHandle::new(&rhs.handle, &rhs.strides, &rhs_original_shape.dims),
        TensorHandle::new(&out.handle, &out.strides, &out.shape.dims),
        Some(UInt::new(D as u32 - 2)),
    );

    out
}
