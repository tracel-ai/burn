use super::init_matmul_output;
use crate::{CubeRuntime, kernel::quantization::dequantize, tensor::CubeTensor};
use burn_backend::{DType, QTensorPrimitive};
use burn_std::QuantLevel;
use cubek::matmul::{
    definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError},
    launch::{MatmulInputHandleRef, Strategy},
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

pub(crate) fn launch_matmul_naive<R: CubeRuntime>(
    strategy: &Strategy,
    mut lhs: CubeTensor<R>,
    mut rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    // Naive has very specific layout requirements for block scaled tensors, so we need to manually
    // dequantize if it fails to launch normally. This is because naive is assumed to always work.
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        match launch_matmul(strategy, lhs.clone(), rhs.clone(), out.clone()) {
            Err(_) => {
                if lhs.qparams.is_some() {
                    lhs = dequantize(lhs, out.dtype);
                }
                if rhs.qparams.is_some() {
                    rhs = dequantize(rhs, out.dtype);
                }
                launch_matmul(strategy, lhs, rhs, out)
            }
            Ok(_) => Ok(()),
        }
    } else {
        launch_matmul(strategy, lhs, rhs, out)
    }
}

pub(crate) fn launch_matmul<R: CubeRuntime>(
    strategy: &Strategy,
    lhs: CubeTensor<R>,
    mut rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), MatmulSetupError> {
    let client = &lhs.client;

    let lhs_quant_handles = lhs.quantized_handles();
    let out_dtype: DType = out.dtype;

    let (lhs_dtype, lhs_handle) = match &lhs_quant_handles {
        None => (
            lhs.dtype,
            MatmulInputHandleRef::new(lhs.as_handle_ref(), lhs.dtype.into()),
        ),
        Some((data, scale)) => (
            out_dtype,
            MatmulInputHandleRef::quantized(
                data.as_handle_ref(),
                scale.as_handle_ref(),
                lhs.meta.shape(),
                lhs.scheme(),
                data.dtype.into(),
                scale.dtype.into(),
            ),
        ),
    };

    let rhs_quant_handles = rhs.quantized_handles();

    let (rhs_dtype, rhs_handle) = match &rhs_quant_handles {
        None => (
            lhs.dtype,
            MatmulInputHandleRef::new(rhs.as_handle_ref(), lhs.dtype.into()),
        ),
        Some((data, scale)) => {
            // Extremely hacky fix to ensure naive can run in every case
            if matches!(strategy, Strategy::Naive)
                && matches!(rhs.scheme().level, QuantLevel::Block(_))
            {
                rhs = dequantize(rhs.clone(), lhs.dtype);
                (
                    lhs.dtype,
                    MatmulInputHandleRef::new(rhs.as_handle_ref(), rhs.dtype.into()),
                )
            } else {
                (
                    out_dtype,
                    MatmulInputHandleRef::quantized(
                        data.as_handle_ref(),
                        scale.as_handle_ref(),
                        rhs.meta.shape(),
                        rhs.scheme(),
                        data.dtype.into(),
                        scale.dtype.into(),
                    ),
                )
            }
        }
    };

    let mut dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: lhs_dtype.into(),
        rhs: rhs_dtype.into(),
        out: out_dtype.into(),
    });
    cubek::matmul::launch::launch_ref(
        strategy,
        client,
        &lhs_handle,
        &rhs_handle,
        &out.as_handle_ref(),
        &mut dtypes,
    )?;

    Ok(())
}
