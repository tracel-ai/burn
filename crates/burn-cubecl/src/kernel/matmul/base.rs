use super::init_matmul_output;
use crate::{CubeRuntime, tensor::CubeTensor};
use burn_backend::{DType, QTensorPrimitive};
use cubek::matmul::{
    MatmulInputHandleRef,
    components::{MatmulElems, MatmulSetupError},
    tune_key::MatmulElemType,
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
pub fn matmul<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    strategy: MatmulStrategy,
    out_dtype: DType,
) -> Result<CubeTensor<R>, MatmulSetupError> {
    match strategy {
        MatmulStrategy::Cube => {
            let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));
            launch_matmul(&Default::default(), lhs, rhs, out.clone())?;
            Ok(out)
        }
        #[cfg(feature = "autotune")]
        MatmulStrategy::Autotune => Ok(matmul_autotune(lhs, rhs, out, out_dtype)),
    }
}

pub(crate) fn launch_matmul<R: CubeRuntime>(
    strategy: &cubek::matmul::Strategy,
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    let client = &lhs.client;
    let mut lhs_quant = false;
    let mut rhs_quant = false;

    let lhs_quant_handles = lhs.quantized_handles();
    let out_dtype: DType = out.dtype;

    let (lhs_dtype, lhs_handle) = match &lhs_quant_handles {
        None => (
            lhs.dtype,
            MatmulInputHandleRef::new(lhs.as_handle_ref(), lhs.dtype.into()),
        ),
        Some((data, scale)) => {
            lhs_quant = true;
            (
                out_dtype,
                MatmulInputHandleRef::quantized(
                    data.as_handle_ref(),
                    scale.as_handle_ref(),
                    &lhs.shape.dims,
                    lhs.scheme(),
                    data.dtype.into(),
                    scale.dtype.into(),
                ),
            )
        }
    };

    let rhs_quant_handles = rhs.quantized_handles();

    let (rhs_dtype, rhs_handle) = match &rhs_quant_handles {
        None => (
            lhs.dtype,
            MatmulInputHandleRef::new(rhs.as_handle_ref(), lhs.dtype.into()),
        ),
        Some((data, scale)) => {
            rhs_quant = true;
            (
                out_dtype,
                MatmulInputHandleRef::quantized(
                    data.as_handle_ref(),
                    scale.as_handle_ref(),
                    &rhs.shape.dims,
                    rhs.scheme(),
                    data.dtype.into(),
                    scale.dtype.into(),
                ),
            )
        }
    };

    let mut dtypes = MatmulElems::from_globals(
        MatmulElemType {
            dtype: lhs_dtype.into(),
            quantized: lhs_quant,
        },
        MatmulElemType {
            dtype: rhs_dtype.into(),
            quantized: rhs_quant,
        },
        MatmulElemType {
            dtype: out_dtype.into(),
            quantized: false,
        },
    );
    cubek::matmul::launch_ref(
        strategy,
        client,
        &lhs_handle,
        &rhs_handle,
        &out.as_handle_ref(),
        &mut dtypes,
    )?;

    Ok(())
}
