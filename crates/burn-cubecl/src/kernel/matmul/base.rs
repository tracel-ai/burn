use super::init_matmul_output;
use crate::{CubeRuntime, FloatElement, tensor::CubeTensor};
use cubecl::linalg::matmul::kernels::MatmulLaunchError;
use cubecl_std::SymQ8;

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
pub fn matmul<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    strategy: MatmulStrategy,
) -> Result<CubeTensor<R>, MatmulLaunchError> {
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
            )?;

            Ok(out)
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => Ok(matmul_autotune::<R, E>(lhs, rhs, out)),
    }
}

/// Launch a quantized matmul kernel using the given strategy.
pub fn q_matmul<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    strategy: MatmulStrategy,
) -> Result<CubeTensor<R>, MatmulLaunchError> {
    // panic!("launch q_matmul");
    let out = out.unwrap_or_else(|| init_matmul_output::<R, half::f16>(&lhs, &rhs));

    let client = &lhs.client;

    cubecl::linalg::matmul::launch_ref::<R, SymQ8>(
        &Default::default(),
        client,
        &lhs.as_handle_ref(),
        &rhs.as_handle_ref(),
        &out.as_handle_ref(),
    )?;

    // panic!("OK");
    Ok(out)
}
