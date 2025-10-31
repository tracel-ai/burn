use super::init_matmul_output;
use crate::{CubeElement, CubeRuntime, tensor::CubeTensor};
use burn_tensor::quantization::QTensorPrimitive;
use cubecl::matmul::{
    MatmulInputHandleRef,
    components::{AccG, MatmulPrecision, MatmulSetupError, MatrixPrecision},
};

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
pub fn matmul<R: CubeRuntime, MP: MatmulPrecision<Acc: MatrixPrecision<Global: CubeElement>>>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    strategy: MatmulStrategy,
) -> Result<CubeTensor<R>, MatmulSetupError> {
    match strategy {
        MatmulStrategy::Cube => {
            let out = out.unwrap_or_else(|| init_matmul_output::<R, AccG<MP>>(&lhs, &rhs));
            launch_matmul::<R, MP>(&Default::default(), lhs, rhs, out.clone())?;
            Ok(out)
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => Ok(matmul_autotune::<R, MP>(lhs, rhs, out)),
    }
}

pub(crate) fn launch_matmul<R: CubeRuntime, MP: MatmulPrecision>(
    strategy: &cubecl::matmul::Strategy,
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    let client = &lhs.client;

    let lhs_quant_handles = lhs.quantized_handles();
    let lhs_handle = match &lhs_quant_handles {
        None => MatmulInputHandleRef::new(lhs.as_handle_ref()),
        Some((data, scale)) => MatmulInputHandleRef::quantized(
            data.as_handle_ref(),
            scale.as_handle_ref(),
            &lhs.shape.dims,
            lhs.scheme(),
        ),
    };

    let rhs_quant_handles = rhs.quantized_handles();
    let rhs_handle = match &rhs_quant_handles {
        None => MatmulInputHandleRef::new(rhs.as_handle_ref()),
        Some((data, scale)) => MatmulInputHandleRef::quantized(
            data.as_handle_ref(),
            scale.as_handle_ref(),
            &rhs.shape.dims,
            rhs.scheme(),
        ),
    };

    cubecl::matmul::launch_ref::<R, MP>(
        strategy,
        client,
        &lhs_handle,
        &rhs_handle,
        &out.as_handle_ref(),
    )?;

    Ok(())
}
