//! A matmul whose output has too few elements to fill the device while its inner dimension `k` is
//! long.
//!
//! The tiled routines split the output, not `k`, so such a problem runs on a few cubes walking all
//! of `k`. Here `k` is cut into chunks: one thread sums each chunk of each output element, one
//! `acc += lhs * rhs` at a time, then one thread per element adds its chunks in order.

use burn_backend::{DType, cubecl::dtype_to_storage_type};
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use crate::{
    kernel::utils::{address_type, chunk_length, device_threads, few_outputs_over_a_long_axis},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};

/// Each thread sums one chunk of `k` for one output element into `partial`, laid out
/// `[batch, chunks, m, n]`.
#[cube(launch, address_type = "dynamic")]
fn matmul_long_k_chunk_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    partial: &mut Tensor<F>,
    size_m: usize,
    size_n: usize,
    chunk: usize,
    chunks: usize,
    #[define(F)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= partial.len() {
        terminate!();
    }
    let rank = lhs.rank();
    let n = ABSOLUTE_POS % size_n;
    let m = (ABSOLUTE_POS / size_n) % size_m;
    let c = (ABSOLUTE_POS / (size_n * size_m)) % chunks;

    let lhs_k = lhs.stride(rank - 1);
    let rhs_k = rhs.stride(rank - 2);
    let size_k = lhs.shape(rank - 1);
    let start = c * chunk;
    let mut batch = ABSOLUTE_POS / (size_n * size_m * chunks);
    let mut lhs_at = m * lhs.stride(rank - 2) + start * lhs_k;
    let mut rhs_at = n * rhs.stride(rank - 1) + start * rhs_k;
    for i in 0..rank - 2 {
        let dim = rank - 3 - i;
        let index = batch % lhs.shape(dim);
        batch /= lhs.shape(dim);
        lhs_at += index * lhs.stride(dim);
        rhs_at += index * rhs.stride(dim);
    }
    let rest = size_k - start;
    let length = if rest < chunk { rest } else { chunk };
    let mut acc = F::new(0.0);
    for _ in 0..length / 4 {
        let a0 = lhs[lhs_at];
        let a1 = lhs[lhs_at + lhs_k];
        let a2 = lhs[lhs_at + 2 * lhs_k];
        let a3 = lhs[lhs_at + 3 * lhs_k];
        let b0 = rhs[rhs_at];
        let b1 = rhs[rhs_at + rhs_k];
        let b2 = rhs[rhs_at + 2 * rhs_k];
        let b3 = rhs[rhs_at + 3 * rhs_k];
        acc += a0 * b0;
        acc += a1 * b1;
        acc += a2 * b2;
        acc += a3 * b3;
        lhs_at += 4 * lhs_k;
        rhs_at += 4 * rhs_k;
    }
    for _ in 0..length % 4 {
        acc += lhs[lhs_at] * rhs[rhs_at];
        lhs_at += lhs_k;
        rhs_at += rhs_k;
    }
    partial[ABSOLUTE_POS] = acc;
}

/// Each thread adds one output element's chunks, in order.
#[cube(launch, address_type = "dynamic")]
fn matmul_long_k_sum_kernel<F: Float>(
    partial: &Tensor<F>,
    out: &mut Tensor<F>,
    chunks: usize,
    #[define(F)] _dtype: ElemType,
) {
    let rank = out.rank();
    if ABSOLUTE_POS >= out.len() {
        terminate!();
    }
    let size_n = out.shape(rank - 1);
    let size_m = out.shape(rank - 2);
    let n = ABSOLUTE_POS % size_n;
    let m = (ABSOLUTE_POS / size_n) % size_m;
    let mut batch = ABSOLUTE_POS / (size_n * size_m);
    let first = batch * chunks * size_m * size_n + m * size_n + n;
    let mut out_at = m * out.stride(rank - 2) + n * out.stride(rank - 1);
    for i in 0..rank - 2 {
        let dim = rank - 3 - i;
        let index = batch % out.shape(dim);
        batch /= out.shape(dim);
        out_at += index * out.stride(dim);
    }
    let mut acc = F::new(0.0);
    for c in 0..chunks {
        acc += partial[first + c * size_m * size_n];
    }
    out[out_at] = acc;
}

/// Whether `out = lhs @ rhs` takes the long-`k` route: an `f32` or `f64` matmul without broadcast
/// batch dimensions whose output is too small to fill the device while its products are not.
pub(crate) fn applies(lhs: &CubeTensor, rhs: &CubeTensor, out: &CubeTensor) -> bool {
    let [lhs_shape, rhs_shape, out_shape] = [lhs, rhs, out].map(|t| t.meta.shape());
    let rank = out_shape.len();
    rank >= 2
        && lhs_shape.len() == rank
        && rhs_shape.len() == rank
        && lhs_shape[..rank - 2] == out_shape[..rank - 2]
        && rhs_shape[..rank - 2] == out_shape[..rank - 2]
        && lhs.qparams.is_none()
        && rhs.qparams.is_none()
        && lhs.dtype == out.dtype
        && rhs.dtype == out.dtype
        // One accumulator per thread over a long k: a half-precision one would overflow.
        && matches!(out.dtype, DType::F32 | DType::F64)
        && few_outputs_over_a_long_axis(
            device_threads(out),
            out_shape.num_elements(),
            lhs_shape[rank - 1],
        )
}

/// `out = lhs @ rhs` by the long-`k` route, for operands it [`applies`] to.
pub(crate) fn matmul_long_k(lhs: CubeTensor, rhs: CubeTensor, out: CubeTensor) {
    let shape = out.meta.shape().clone();
    let rank = shape.len();
    let size_k = lhs.meta.shape()[rank - 1];
    let outputs = shape.num_elements();
    let chunk = chunk_length(size_k);
    let chunks = size_k.div_ceil(chunk);
    let client = out.client.clone();
    let dtype = out.dtype;
    let partial = empty_device_dtype(
        client.clone(),
        out.device.clone(),
        [outputs * chunks].into(),
        dtype,
    );
    let cube_dim = CubeDim::new(&client, outputs * chunks);
    matmul_long_k_chunk_kernel::launch(
        &client,
        calculate_cube_count_elemwise(&client, outputs * chunks, cube_dim),
        cube_dim,
        address_type!(lhs, rhs, partial),
        lhs.into_tensor_arg(),
        rhs.into_tensor_arg(),
        partial.clone().into_tensor_arg(),
        shape[rank - 2],
        shape[rank - 1],
        chunk,
        chunks,
        dtype_to_storage_type(dtype),
    );
    // A plane per cube spreads the few threads over as many cores as there are planes.
    let cube_dim = CubeDim::new_1d(client.properties().hardware.plane_size_max);
    matmul_long_k_sum_kernel::launch(
        &client,
        calculate_cube_count_elemwise(&client, outputs, cube_dim),
        cube_dim,
        address_type!(partial, out),
        partial.into_tensor_arg(),
        out.into_tensor_arg(),
        chunks,
        dtype_to_storage_type(dtype),
    );
}
