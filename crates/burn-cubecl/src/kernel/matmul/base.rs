use super::init_matmul_output;
use crate::{CubeRuntime, element::MatmulElement, tensor::CubeTensor};
use burn_tensor::{DType, quantization::QuantAcc};
use cubecl::matmul::{
    MatmulInputHandleRef,
    components::{AccG, MatmulSetupError},
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
pub fn matmul<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    strategy: MatmulStrategy,
) -> Result<CubeTensor<R>, MatmulSetupError> {
    match strategy {
        MatmulStrategy::Cube => {
            let out = out.unwrap_or_else(|| init_matmul_output::<R, AccG<E>>(&lhs, &rhs));

            let client = &lhs.client;

            cubecl::matmul::launch_ref::<R, E>(
                &Default::default(),
                client,
                &MatmulInputHandleRef::Normal(lhs.as_handle_ref()),
                &MatmulInputHandleRef::Normal(rhs.as_handle_ref()),
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

    lhs.dtype = DType::I8;
    rhs.dtype = DType::I8;

    let lhs_scales = lhs.scales().unwrap();
    let rhs_scales = rhs.scales().unwrap();

    match QuantAcc::default() {
        QuantAcc::F32 => {
            cubecl::matmul::launch_ref::<R, (i8, i8, half::f16, half::f16, f32, half::f16)>(
                &Default::default(),
                client,
                &MatmulInputHandleRef::Quantized {
                    data: lhs.as_handle_ref(),
                    scale: lhs_scales.as_handle_ref(),
                },
                &MatmulInputHandleRef::Quantized {
                    data: rhs.as_handle_ref(),
                    scale: rhs_scales.as_handle_ref(),
                },
                &out.as_handle_ref(),
            )?;
        }
        QuantAcc::F16 => {
            cubecl::matmul::launch_ref::<R, (i8, i8, half::f16, half::f16, half::f16, half::f16)>(
                &Default::default(),
                client,
                &MatmulInputHandleRef::Quantized {
                    data: lhs.as_handle_ref(),
                    scale: lhs_scales.as_handle_ref(),
                },
                &MatmulInputHandleRef::Quantized {
                    data: rhs.as_handle_ref(),
                    scale: rhs_scales.as_handle_ref(),
                },
                &out.as_handle_ref(),
            )?;
        }
        QuantAcc::BF16 => unimplemented!(),
    }

    Ok(out)
}
