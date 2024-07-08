use std::cmp::max;

use burn_cube::{frontend::TensorArg, Compiler};

use crate::{
    kernel::{
        into_contiguous,
        matmul::{
            config::{tiling2d_cube_count, tiling2d_cube_dim, CubeTiling2dConfig},
            Tiling2dConfig,
        },
    },
    tensor::{JitTensor, MemoryLayout},
    FloatElement, JitRuntime,
};

use super::base::tiling2d_cube_launch;

/// Matrix multiplication using tiling 2d algorithm
pub fn matmul_tiling_2d_cube<'a, R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    assert!(
        config.block_size_k * max(config.block_size_m, config.block_size_n)
            <= <R::Compiler as Compiler>::max_shared_memory_size(),
        "Shared memory limit will be busted. "
    );

    let m = lhs.shape.dims[D - 2];
    let k = lhs.shape.dims[D - 1];
    let n = rhs.shape.dims[D - 1];

    let client = lhs.client.clone();

    let check_layout = |tensor: JitTensor<R, E, D>| match tensor.memory_layout() {
        MemoryLayout::Contiguous => (tensor, false),
        MemoryLayout::MildlyPermuted {
            transposed,
            batch_swap: _,
        } => (tensor, transposed),
        MemoryLayout::HighlyPermuted => (into_contiguous(tensor), false),
    };
    let (lhs, lhs_transposed) = check_layout(lhs);
    let (rhs, rhs_transposed) = check_layout(rhs);

    let vectorization = |shape: usize| {
        [4, 2]
            .into_iter()
            .filter(|v| shape % v == 0)
            .map(|v| v as u8)
            .next()
            .unwrap_or(1)
    };

    let lhs_vectorization = match lhs_transposed {
        true => vectorization(m),
        false => 1,
    };
    let rhs_vectorization = match rhs_transposed {
        true => 1,
        false => vectorization(n),
    };
    let out_vectorization = vectorization(n);

    let cube_count = tiling2d_cube_count::<R, D>(&out.shape, &config);
    let cube_dim = tiling2d_cube_dim(&config);
    let cube_config = CubeTiling2dConfig::new(&config, m, k, n, lhs_transposed, rhs_transposed);

    tiling2d_cube_launch::<E::FloatPrimitive, R>(
        client,
        cube_count,
        cube_dim,
        TensorArg::vectorized(
            lhs_vectorization,
            &lhs.handle,
            &lhs.strides,
            &lhs.shape.dims,
        ),
        TensorArg::vectorized(
            rhs_vectorization,
            &rhs.handle,
            &rhs.strides,
            &rhs.shape.dims,
        ),
        TensorArg::vectorized(
            out_vectorization,
            &out.handle,
            &out.strides,
            &out.shape.dims,
        ),
        cube_config,
    );

    out
}
