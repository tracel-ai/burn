//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use crate::{
    kernel::{into_contiguous, SUBCUBE_DIM_APPROX},
    ops::swap_dims,
    tensor::JitTensor,
    FloatElement, JitRuntime,
};

use super::simple_cube_count;
use cubecl::prelude::*;

#[cube(launch_unchecked)]
fn matmul_kernel<F: Float>(
    lhs: &Tensor<F>,
    rhs: &Tensor<F>,
    out: &mut Tensor<F>,
    // number of dimensions not involved in the matmul
    #[comptime] num_batches: Option<u32>,
) {
    let rank = out.rank();
    let end = num_batches.unwrap_or_else(|| rank - 2);
    let unroll = num_batches.is_some();

    let n_rows = lhs.shape(rank - 2);
    let n_cols = rhs.shape(rank - 1);
    let mut k = rhs.shape(rank - 2);

    let batch_pos = ABSOLUTE_POS_Z;
    let row = CUBE_DIM_X * CUBE_POS_X + UNIT_POS_X;
    let col = CUBE_DIM_Y * CUBE_POS_Y + UNIT_POS_Y;

    if row >= n_rows || col >= n_cols {
        return;
    }

    let vectorization_factor = vectorization_of(lhs);

    let mut offset_lhs = 0;
    let mut offset_rhs = 0;
    let offset_out = n_rows * n_cols * batch_pos;

    #[unroll(unroll)]
    for i in 0..end {
        let ogwl = offset_out / out.stride(i);

        offset_lhs += ogwl % lhs.shape(i) * lhs.stride(i);
        offset_rhs += ogwl % rhs.shape(i) * rhs.stride(i);
    }

    offset_lhs /= vectorization_factor;
    offset_rhs /= vectorization_factor;

    let mut sum = F::vectorized(0., vectorization_factor);

    k /= vectorization_factor;

    for i in 0..k {
        let lhs_index = row * k + i + offset_lhs;
        let rhs_index = col * k + i + offset_rhs;

        sum += lhs[lhs_index] * rhs[rhs_index];
    }

    let mut out_index = row * n_cols + col;
    out_index += offset_out;

    let unroll_sum = vectorization_factor != 1;
    if unroll_sum {
        let mut accum = F::new(0.);
        // we unroll the loop to sum `vectorization_factor` elements at once, which lets us
        // use SIMD instructions to speed up the computation
        #[unroll]
        for v in 0..vectorization_factor {
            accum += sum[v];
        }

        out[out_index] = accum;
    } else {
        out[out_index] = sum;
    }
}

/// Matrix multiplication using memory coalescing algorithm with cube dimensions of size 16
pub fn matmul_mem_coalescing_default<R: JitRuntime, E: FloatElement>(
    lhs: JitTensor<R, E>,
    rhs: JitTensor<R, E>,
    out: JitTensor<R, E>,
) -> JitTensor<R, E> {
    matmul_simple::<R, E>(lhs, rhs, out, SUBCUBE_DIM_APPROX, SUBCUBE_DIM_APPROX)
}

/// Matrix multiplication using memory coalescing algorithm with custom cube dimensions
pub fn matmul_simple<R: JitRuntime, E: FloatElement>(
    lhs: JitTensor<R, E>,
    rhs: JitTensor<R, E>,
    out: JitTensor<R, E>,
    cube_dim_x: usize,
    cube_dim_y: usize,
) -> JitTensor<R, E> {
    lhs.assert_is_on_same_device(&rhs);
    let ndims = lhs.shape.num_dims();
    let lhs = into_contiguous(lhs);

    let rhs_original_shape = rhs.shape.clone();
    // we swap the dimensions to achieve memory-coalescing:
    // consecutive elements of a column in the original rhs tensor will now be stored
    // consecutively in memory, which allows to fetch them with fewer memory instructions
    let rhs = into_contiguous(swap_dims(rhs, ndims - 1, ndims - 2));

    let cube_count = simple_cube_count::<R>(
        &lhs.shape,
        &rhs_original_shape,
        &out.shape,
        cube_dim_x,
        cube_dim_y,
    );

    let vectorization_factor = match lhs.shape.dims[ndims - 1] % 4 == 0 {
        true => 4,
        false => 1,
    };

    unsafe {
        matmul_kernel::launch_unchecked::<E, R>(
            &lhs.client,
            cube_count,
            CubeDim::new(cube_dim_x as u32, cube_dim_y as u32, 1),
            lhs.as_tensor_arg(vectorization_factor),
            TensorArg::from_raw_parts(
                &rhs.handle,
                &rhs.strides,
                &rhs_original_shape.dims, // We need the original shape.
                vectorization_factor,
            ),
            out.as_tensor_arg(1),
            Some(ndims as u32 - 2),
        );
    };

    out
}
