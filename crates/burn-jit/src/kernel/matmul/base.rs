use crate::{tensor::JitTensor, JitElement, Runtime};

use super::{
    init_matmul_output, matmul_autotune, matmul_mem_coalescing,
    unpadded::matmul_tiling_2d_unpadded, vec4::matmul_tiling_2d_vec4,
};

/// The strategy to be used when launching a matmul kernel.
#[derive(Default)]
pub enum MatmulStrategy {
    /// A simple kernel will be used with memory coalescing optimization.
    Simple {
        /// Grad size x
        grid_x: usize,
        /// Grad size y
        grid_y: usize,
    },
    /// A tiling 2d kernel will be used, with support for any matrix size without padding.
    Tiling2d,
    /// A tiling 2d kernel will be used, with support for any matrix size with padding.
    Tiling2dPadded,
    #[cfg(feature = "autotune")]
    /// Using autotune to chose the best kernel based on runtime information.
    #[default]
    Autotune,
}

#[cfg(feature = "autotune")]
#[cfg(not(feature = "autotune"))]
impl Default for MatmulStrategy {
    fn default() -> Self {
        MatmulStrategy::Tiling2d
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    strategy: MatmulStrategy,
) -> JitTensor<R, E, D> {
    match strategy {
        MatmulStrategy::Simple { grid_x, grid_y } => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_mem_coalescing(lhs, rhs, out, grid_x, grid_y)
        }
        MatmulStrategy::Tiling2d => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_tiling_2d_unpadded(lhs, rhs, out)
        }
        MatmulStrategy::Tiling2dPadded => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_tiling_2d_vec4(lhs, rhs, out)
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => matmul_autotune(lhs, rhs),
    }
}
