use super::init_matmul_output;
use crate::{tensor::JitTensor, FloatElement, JitRuntime};

#[cfg(feature = "autotune")]
use super::matmul_autotune;

/// The strategy to be used when launching a matmul kernel.
pub enum MatmulStrategy {
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
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
    out: Option<JitTensor<R>>,
    strategy: MatmulStrategy,
) -> JitTensor<R> {
    match strategy {
        MatmulStrategy::Cube => {
            let out = out.unwrap_or_else(|| init_matmul_output::<R, E>(&lhs, &rhs));

            let client = &lhs.client;

            cubecl::linalg::matmul::launch_ref::<R, E>(
                &Default::default(),
                client,
                &lhs.as_handle_ref(),
                &rhs.as_handle_ref(),
                &out.as_handle_ref(),
            );
            out
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => matmul_autotune::<R, E>(lhs, rhs, out),
    }
}
