use burn_tensor::Element;
use cubecl::{
    matmul::{
        Strategy, SyncBufferLoadingStrategy, SyncLoadingStrategy,
        components::MatmulKind,
        kernels::matmul::{
            Selection, double_buffering::DoubleBufferingArgs,
            ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
        },
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
        .with_tunable_optional(naive::<R, E>, |key| {
            matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
                || matches!(key.analysis.kind, MatmulKind::General)
        })
        .with_tunable(simple_cube::<R, E>)
        .with_tunable_optional(matmul_simple::<R, E>, |key| {
            matches!(key.analysis.kind, MatmulKind::General)
        })
        .with_tunable_optional(matmul_simple_multi_rows::<R, E>, |key| {
            matches!(key.analysis.kind, MatmulKind::General)
        })
        .with_tunable_optional(matmul_ordered_double_buffering_1::<R, E>, |key| {
            matches!(key.analysis.kind, MatmulKind::General)
        })
        .with_tunable_optional(matmul_ordered_double_buffering_2::<R, E>, |key| {
            matches!(key.analysis.kind, MatmulKind::General)
        })
        .with_tunable_optional(matmul_double_buffering_specialized::<R, E>, |key| {
            should_tune_double_buffering(false, key)
        })
        .with_tunable_optional(matmul_double_buffering::<R, E>, |key| {
            should_tune_double_buffering(false, key)
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
    cubecl::matmul::launch_ref::<R, E>(
        &Strategy::Simple(
            SyncLoadingStrategy::Cyclic,
            Selection::Inferred(SimpleArgs { multi_rows: false }),
        ),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_simple_multi_rows<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::matmul::launch_ref::<R, E>(
        &Strategy::Simple(
            SyncLoadingStrategy::Cyclic,
            Selection::Inferred(SimpleArgs { multi_rows: true }),
        ),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_double_buffering<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::matmul::launch_ref::<R, E>(
        &Strategy::DoubleBuffering(
            SyncBufferLoadingStrategy::Tilewise,
            Selection::Inferred(DoubleBufferingArgs { specialized: false }),
        ),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_double_buffering_specialized<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::matmul::launch_ref::<R, E>(
        &Strategy::DoubleBuffering(
            SyncBufferLoadingStrategy::Tilewise,
            Selection::Inferred(DoubleBufferingArgs { specialized: true }),
        ),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_ordered_double_buffering_1<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::matmul::launch_ref::<R, E>(
        &Strategy::OrderedDoubleBuffering(Selection::Inferred(OrderedSelectionArgs {
            partition_k: Some(16),
            row_count: Some(2),
            rows_per_plane: Some(1),
        })),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_ordered_double_buffering_2<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::matmul::launch_ref::<R, E>(
        &Strategy::OrderedDoubleBuffering(Selection::Inferred(OrderedSelectionArgs {
            partition_k: Some(8),
            row_count: Some(2),
            rows_per_plane: Some(2),
        })),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn simple_cube<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::matmul::launch_ref::<R, E>(
        &Strategy::SimpleUnit(None),
        &lhs.client,
        &lhs.as_handle_ref(),
        &None,
        &rhs.as_handle_ref(),
        &None,
        &out.as_handle_ref(),
    )
    .map_err(|err| format!("{err:?}"))
}

fn naive<R: CubeRuntime, E: FloatElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    cubecl::matmul::launch_ref::<R, E>(
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
