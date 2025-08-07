use super::init_matmul_output;
use crate::{CubeRuntime, FloatElement, tensor::CubeTensor};
use burn_tensor::{
    DType,
    quantization::{QTensorPrimitive, QuantFloatPrecision},
};
use cubecl::matmul::components::{MatmulSetupError, Quantized};

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
) -> Result<CubeTensor<R>, MatmulSetupError> {
    match strategy {
        MatmulStrategy::Cube => {
            let out = out.unwrap_or_else(|| init_matmul_output::<R, E>(&lhs, &rhs));

            let client = &lhs.client;

            cubecl::matmul::launch_ref::<R, E>(
                &Default::default(),
                client,
                &lhs.as_handle_ref(),
                &None,
                &rhs.as_handle_ref(),
                &None,
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
    mut lhs: CubeTensor<R>,
    mut rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    _strategy: MatmulStrategy,
) -> Result<CubeTensor<R>, MatmulSetupError> {
    let out = out.unwrap_or_else(|| init_matmul_output::<R, half::f16>(&lhs, &rhs));

    let client = &lhs.client;

    let scheme = *lhs.scheme();

    lhs.dtype = DType::I8;
    rhs.dtype = DType::I8;

    let lhs_scales = lhs.scales().unwrap();
    let rhs_scales = rhs.scales().unwrap();

    match scheme.acc_precision {
        QuantFloatPrecision::F32 => {
            cubecl::matmul::launch_ref::<R, (i8, half::f16, f32, half::f16, Quantized)>(
                &Default::default(),
                client,
                &lhs.as_handle_ref(),
                &Some(lhs_scales.as_handle_ref()),
                &rhs.as_handle_ref(),
                &Some(rhs_scales.as_handle_ref()),
                &out.as_handle_ref(),
            )?;
        }
        QuantFloatPrecision::F16 => {
            cubecl::matmul::launch_ref::<R, (i8, half::f16, half::f16, half::f16, Quantized)>(
                &Default::default(),
                client,
                &lhs.as_handle_ref(),
                &Some(lhs_scales.as_handle_ref()),
                &rhs.as_handle_ref(),
                &Some(rhs_scales.as_handle_ref()),
                &out.as_handle_ref(),
            )?;
        }
        QuantFloatPrecision::BF16 => unimplemented!(),
    }

    Ok(out)
}
