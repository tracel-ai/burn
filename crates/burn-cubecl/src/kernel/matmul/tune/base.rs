use burn_tensor::Element;
use cubecl::{
    linalg::matmul::{
        Strategy, SyncLoadingStrategy,
        components::MatmulKind,
        kernels::tiling2d::Tiling2dConfig,
        tune_key::{MatmulAutotuneKey, MatmulGlobalScale, should_tune_double_buffering},
    },
    tune::{LocalTuner, TunableSet, local_tuner},
};

use crate::{
    CubeRuntime, CubeTuneId, element::FloatElement, kernel::matmul::utils::init_matmul_output,
    tensor::CubeTensor,
};

fn matmul_input_gen<R: CubeRuntime>(
    _key: &MatmulAutotuneKey,
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
    out: &CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    (lhs.clone(), rhs.clone(), out.copy())
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<R: CubeRuntime, E: FloatElement + Element>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output::<R, E>(&lhs, &rhs));

    let client = lhs.client.clone();

    static TUNER: LocalTuner<MatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R>, matmul_input_gen::<R>)
        .with_tunable_optional(matmul_tiling2d::<R, E>, |key| {
            !key.analysis.may_use_tensor_cores
                || matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
        })
        .with_tunable(matmul_simple::<R, E>)
        .with_tunable_optional(matmul_double_buffering::<R, E>, |key| {
            should_tune_double_buffering(false, key)
        })
        .with_tunable_optional(matmul_naive::<R, E>, |key| {
            !key.analysis.may_use_tensor_cores
                || !matches!(
                    key.analysis.kind,
                    MatmulKind::OuterProduct | MatmulKind::General
                )
        });

    TUNER.execute(
        &CubeTuneId::new::<R>(&lhs.client, &lhs.device),
        &client,
        &tunables,
        (lhs, rhs, output.clone()),
    );

    output
}

fn create_key<R: CubeRuntime>(
    lhs: &CubeTensor<R>,
    rhs: &CubeTensor<R>,
    out: &CubeTensor<R>,
) -> MatmulAutotuneKey {
    MatmulAutotuneKey::generate::<R>(
        &lhs.client,
        &lhs.shape.dims,
        &rhs.shape.dims,
        &lhs.strides,
        &rhs.strides,
        lhs.dtype.into(),
        rhs.dtype.into(),
        out.dtype.into(),
    )
}

fn matmul_simple<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Simple(SyncLoadingStrategy::Cyclic),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

// Creates invalid configs for some shapes, re-enable once fixed
fn matmul_double_buffering<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::DoubleBuffering(cubecl::linalg::matmul::SyncBufferLoadingStrategy::Cyclic),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_tiling2d<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Tiling2D(Tiling2dConfig::default()),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_naive<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::linalg::matmul::launch_ref::<R, E>(
        &Strategy::Naive,
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}
