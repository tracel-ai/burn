use crate::{tensor::JitTensor, FloatElement, JitRuntime};
use burn_cube::{prelude::*, Compiler};
use burn_tensor::Shape;
use std::cmp::{max, min};

use super::{
    init_matmul_output, matmul_autotune, matmul_simple, matmul_tiling_2d, matmul_tiling_2d_padded,
};

#[derive(Debug, Clone)]
/// Tiling 2D parameters
pub struct Tiling2dConfig {
    /// Number of invocations in x
    pub grid_x: usize,
    /// Number of invocations in y
    pub grid_y: usize,
    /// Block size along dimension of lhs
    pub block_size_m: usize,
    /// Block size along common dimension
    pub block_size_k: usize,
    /// Block size along dimension of rhs
    pub block_size_n: usize,
    /// Tile size along dimension of lhs
    pub tile_size_m: usize,
    /// Tile size along dimension of rhs
    pub tile_size_n: usize,
    /// Loop unrolling
    pub unroll: bool,
}

impl Tiling2dConfig {
    #[allow(unused, clippy::too_many_arguments)]
    fn new<R: JitRuntime>(
        grid_x: usize,
        grid_y: usize,
        block_size_m: usize,
        block_size_k: usize,
        block_size_n: usize,
        tile_size_m: usize,
        tile_size_n: usize,
        unroll: bool,
    ) -> Self {
        assert!(grid_x == f32::ceil(block_size_m as f32 / tile_size_m as f32) as usize);
        assert!(grid_y == f32::ceil(block_size_n as f32 / tile_size_n as f32) as usize);
        assert!(
            block_size_k <= min(block_size_m, block_size_n),
            "Not enough invocations to fill shared memory"
        );
        assert!(
            block_size_k * max(block_size_m, block_size_n)
                <= <R::Compiler as Compiler>::max_shared_memory_size(),
            "Shared memory limit will be busted. "
        );
        assert!(
            block_size_m % tile_size_m == 0 && block_size_n % tile_size_n == 0,
            "Tile size must divide block size in m and n dimensions"
        );
        Self {
            grid_x,
            grid_y,
            block_size_m,
            block_size_k,
            block_size_n,
            tile_size_m,
            tile_size_n,
            unroll,
        }
    }
}

impl Default for Tiling2dConfig {
    fn default() -> Self {
        Self {
            grid_x: 16,
            grid_y: 16,
            block_size_m: 64,
            block_size_k: 32,
            block_size_n: 64,
            tile_size_m: 4,
            tile_size_n: 4,
            unroll: false,
        }
    }
}

/// The strategy to be used when launching a matmul kernel.
pub enum MatmulStrategy {
    /// A simple kernel will be used with memory coalescing optimization.
    Simple {
        /// Number of invocations in x
        grid_x: usize,
        /// Number of invocations in y
        grid_y: usize,
    },
    /// A tiling 2d kernel will be used, with support for any matrix size without padding.
    Tiling2d(Tiling2dConfig),
    /// A tiling 2d kernel will be used, with support for any matrix size with padding.
    Tiling2dPadded(Tiling2dConfig),
    #[cfg(feature = "autotune")]
    /// Using autotune to chose the best kernel based on runtime information.
    Autotune,
}

#[allow(clippy::derivable_impls)] // Necessary otherwise the feature flags dont' work.
#[cfg(feature = "autotune")]
impl Default for MatmulStrategy {
    fn default() -> Self {
        MatmulStrategy::Autotune
    }
}

#[cfg(not(feature = "autotune"))]
impl Default for MatmulStrategy {
    fn default() -> Self {
        MatmulStrategy::Tiling2d(Tiling2dConfig::default())
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul<R: JitRuntime, E: FloatElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    strategy: MatmulStrategy,
) -> JitTensor<R, E, D> {
    match strategy {
        MatmulStrategy::Simple { grid_x, grid_y } => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_simple(lhs, rhs, out, grid_x, grid_y)
        }
        MatmulStrategy::Tiling2d(config) => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_tiling_2d(lhs, rhs, out, config)
        }
        MatmulStrategy::Tiling2dPadded(config) => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_tiling_2d_padded(lhs, rhs, out, config)
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => matmul_autotune(lhs, rhs),
    }
}

pub(crate) fn simple_launch_options<const D: usize>(
    lhs_shape: &Shape<D>,
    rhs_shape: &Shape<D>,
    output_shape: &Shape<D>,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> CubeCount {
    let num_rows = lhs_shape.dims[D - 2];
    let num_cols = rhs_shape.dims[D - 1];

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / workgroup_size_x as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / workgroup_size_y as f32) as u32;
    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output_shape.dims[i];
    }

    CubeCount::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32)
}

pub(crate) fn tiling2d_launch_options<const D: usize>(
    output_shape: &Shape<D>,
    config: Tiling2dConfig,
) -> CubeCount {
    let num_rows = output_shape.dims[D - 2];
    let num_cols = output_shape.dims[D - 1];

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / config.block_size_m as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / config.block_size_n as f32) as u32;
    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output_shape.dims[i];
    }

    CubeCount::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32)
}
