use crate::{tensor::JitTensor, FloatElement, JitRuntime};
use burn_cube::prelude::*;
use burn_tensor::Shape;

use super::{
    config::Tiling2dConfig, init_matmul_output, matmul_autotune, matmul_simple, matmul_tiling_2d,
    matmul_tiling_2d_cube, matmul_tiling_2d_padded,
};

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
    /// A tiling 2d kernel with everything vectorized, and comptime bound checks
    Tiling2dCube(Tiling2dConfig),
}

#[allow(clippy::derivable_impls)] // Necessary otherwise the feature flags dont' work.
#[cfg(feature = "autotune")]
impl Default for MatmulStrategy {
    fn default() -> Self {
        MatmulStrategy::Tiling2dCube(Tiling2dConfig::default())
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
        MatmulStrategy::Tiling2dCube(config) => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_tiling_2d_cube(lhs, rhs, out, config)
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
