use super::{init_matmul_output, tile};
use crate::tensor::CubeTensor;
use burn_backend::DType;
use burn_std::{MatmulTransformAnalysis, MatmulTransformPolicy};
use cubek::matmul::{
    definition::MatmulSetupError,
    kernels::{MatmulKernelHeuristic, MatmulOpStrategy},
};

/// The strategy to be used when launching a matmul kernel. Either way the product runs on
/// cubek's tile-DSL routines.
pub enum MatmulStrategy {
    #[cfg(feature = "autotune")]
    /// The routine measured fastest for the problem on this device.
    Autotune,
    /// The routine cubek's rules pick, without a tuning round.
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

impl From<MatmulStrategy> for MatmulOpStrategy {
    fn from(strategy: MatmulStrategy) -> Self {
        match strategy {
            #[cfg(feature = "autotune")]
            MatmulStrategy::Autotune => MatmulOpStrategy::Autotune,
            MatmulStrategy::Cube => MatmulOpStrategy::Heuristic(MatmulKernelHeuristic::Tile),
        }
    }
}

/// Launch a matmul kernel using the given strategy.
pub fn matmul(
    mut lhs: CubeTensor,
    rhs: CubeTensor,
    out: Option<CubeTensor>,
    strategy: MatmulStrategy,
    out_dtype: DType,
) -> Result<CubeTensor, MatmulSetupError> {
    let out = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));

    // A broadcast-rhs batched matmul that would tile poorly is folded into a
    // single matmul: `[.., b, m, k] @ [.., 1, k, n]` runs as `[.., 1, b*m, k]`
    // instead of `b` matmuls that each re-read the whole rhs. Pure metadata —
    // the launch operands share the handles, the returned tensor keeps the
    // broadcast shape. The analysis reads row strides, which a storage-tiled
    // operand does not have.
    let mut out_launch = out.clone();
    if lhs.qparams.is_none() && !lhs.meta.is_tiled() && !rhs.meta.is_tiled() {
        let analysis = MatmulTransformAnalysis::from_metadata(&lhs.meta, &rhs.meta, &out.meta);
        let action = MatmulTransformPolicy::default().action(&analysis);
        action.apply(&mut lhs.meta);
        action.apply(&mut out_launch.meta);
    }

    tile::launch(lhs, rhs, out_launch, out_dtype, strategy.into())?;
    Ok(out)
}
