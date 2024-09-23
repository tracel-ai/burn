use super::{init_matmul_output, matmul_simple};
use crate::{tensor::JitTensor, FloatElement, JitRuntime};
use burn_tensor::Shape;
use cubecl::prelude::*;

#[cfg(feature = "autotune")]
use super::matmul_autotune;

/// The strategy to be used when launching a matmul kernel.
pub enum MatmulStrategy {
    /// A simple kernel will be used with memory coalescing optimization.
    Simple {
        /// Number of invocations in x
        grid_x: usize,
        /// Number of invocations in y
        grid_y: usize,
    },
    #[cfg(feature = "autotune")]
    /// Using autotune to choose the best kernel based on runtime information.
    Autotune,
    /// Cube implementation of matmul.
    Cube,
}

impl Default for MatmulStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return MatmulStrategy::Autotune;

        #[cfg(not(feature = "autotune"))]
        MatmulStrategy::Cube
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul<R: JitRuntime, E: FloatElement>(
    lhs: JitTensor<R, E>,
    rhs: JitTensor<R, E>,
    strategy: MatmulStrategy,
) -> JitTensor<R, E> {
    match strategy {
        MatmulStrategy::Simple { grid_x, grid_y } => {
            let out = init_matmul_output(&lhs, &rhs);
            matmul_simple(lhs, rhs, out, grid_x, grid_y)
        }
        MatmulStrategy::Cube => {
            let out = init_matmul_output::<R, E>(&lhs, &rhs);
            let client = &lhs.client;
            cubecl::linalg::matmul::launch_ref::<R, E>(
                client,
                lhs.as_handle_ref(),
                rhs.as_handle_ref(),
                out.as_handle_ref(),
            );
            out
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => matmul_autotune(lhs, rhs),
    }
}

pub(crate) fn simple_cube_count<R: JitRuntime>(
    lhs_shape: &Shape,
    rhs_shape: &Shape,
    output_shape: &Shape,
    cube_dim_x: usize,
    cube_dim_y: usize,
) -> CubeCount<R::Server> {
    let ndims = lhs_shape.num_dims();
    let num_rows = lhs_shape.dims[ndims - 2];
    let num_cols = rhs_shape.dims[ndims - 1];

    let cubes_x = f32::ceil(num_rows as f32 / cube_dim_x as f32) as u32;
    let cubes_y = f32::ceil(num_cols as f32 / cube_dim_y as f32) as u32;
    let mut num_iter = 1;
    for i in 0..ndims - 2 {
        num_iter *= output_shape.dims[i];
    }

    CubeCount::Static(cubes_x, cubes_y, num_iter as u32)
}
