use burn_tensor::DType;
use cubecl::{
    matmul::{
        Strategy, SyncPartialReadingStrategy, SyncReadingStrategy,
        components::{AccG, MatmulKind},
        kernels::layered::{
            Selection, TileSizeSelection, double_buffering::DoubleBufferingArgs,
            ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
            simple_unit::SimpleUnitSelectionArgs,
        },
        tune_key::{MatmulAutotuneKey, MatmulGlobalScale, should_tune_double_buffering},
    },
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};

use crate::{
    CubeRuntime, CubeTuneId,
    element::MatmulElement,
    kernel::matmul::{launch_matmul, utils::init_matmul_output},
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
pub fn matmul_autotune<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output::<R, AccG<E>>(&lhs, &rhs));

    let client = lhs.client.clone();

    static TUNER: LocalTuner<MatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: u8 = 3;
        const PRIORITY_HIGH: u8 = 2;
        const PRIORITY_MEDIUM: u8 = 1;
        const PRIORITY_MIN: u8 = 0;

        let cmma = TuneGroup::<MatmulAutotuneKey>::new(|key| {
            if matches!(key.analysis.kind, MatmulKind::General) {
                PRIORITY_MAX
            } else {
                PRIORITY_MEDIUM
            }
        });

        let odd = TuneGroup::<MatmulAutotuneKey>::new(|key| {
            if key.definition.lhs_pow2_factor == 0 || key.definition.rhs_pow2_factor == 0 {
                PRIORITY_MAX
            } else {
                PRIORITY_MIN
            }
        });

        let unit = TuneGroup::<MatmulAutotuneKey>::new(|key| {
            if !matches!(key.analysis.kind, MatmulKind::General)
                || matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
            {
                PRIORITY_MAX
            } else {
                PRIORITY_MIN
            }
        });

        fn double_buffering_priority(key: &MatmulAutotuneKey, max: u8, min: u8) -> u8 {
            if should_tune_double_buffering(false, key) {
                max
            } else {
                min
            }
        }

        TunableSet::new(create_key::<R>, matmul_input_gen::<R>)
            .with(Tunable::new(naive::<R, E>).group(&unit, |key| {
                if matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
                    || matches!(key.analysis.kind, MatmulKind::InnerProduct)
                {
                    PRIORITY_MAX
                } else {
                    PRIORITY_MIN
                }
            }))
            .with(Tunable::new(simple_unit_min::<R, E>).group(&unit, |key| {
                if matches!(key.analysis.kind, MatmulKind::General)
                    && matches!(key.analysis.scale_global, MatmulGlobalScale::Large)
                {
                    PRIORITY_MAX
                } else {
                    PRIORITY_HIGH
                }
            }))
            .with(Tunable::new(simple_unit_max::<R, E>).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(simple_vec_mat::<R, E>).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(double_vec_mat::<R, E>).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(double_unit::<R, E>).group(&unit, |key| {
                double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH)
            }))
            .with(Tunable::new(matmul_simple::<R, E>).group(&cmma, |_| PRIORITY_MAX))
            .with(Tunable::new(matmul_simple_multi_rows::<R, E>).group(&cmma, |_| PRIORITY_MAX))
            .with(
                // Ordered should be tried most of the time.
                Tunable::new(matmul_ordered_double_buffering::<R, E>)
                    .group(&cmma, |_| PRIORITY_MAX),
            )
            .with(
                Tunable::new(matmul_double_buffering_specialized::<R, E>)
                    .group(&cmma, |key| {
                        double_buffering_priority(key, PRIORITY_HIGH, PRIORITY_MEDIUM)
                    })
                    .group(&odd, |_| PRIORITY_MAX),
            )
            .with(
                Tunable::new(matmul_double_buffering::<R, E>)
                    .group(&cmma, |key| {
                        double_buffering_priority(key, PRIORITY_HIGH, PRIORITY_MEDIUM)
                    })
                    .group(&odd, |_| PRIORITY_MAX),
            )
    });

    TUNER.execute(
        &CubeTuneId::new::<R>(&lhs.client, &lhs.device),
        &client,
        tunables,
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

fn matmul_simple<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(
        &Strategy::Simple(
            SyncReadingStrategy::Cyclic,
            Selection::Inferred(SimpleArgs { multi_rows: false }),
        ),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_simple_multi_rows<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(
        &Strategy::Simple(
            SyncReadingStrategy::Cyclic,
            Selection::Inferred(SimpleArgs { multi_rows: true }),
        ),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_double_buffering<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(
        &Strategy::DoubleBuffering(
            SyncPartialReadingStrategy::Tilewise,
            Selection::Inferred(DoubleBufferingArgs { specialized: false }),
        ),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_double_buffering_specialized<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(
        &Strategy::DoubleBuffering(
            SyncPartialReadingStrategy::Tilewise,
            Selection::Inferred(DoubleBufferingArgs { specialized: true }),
        ),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_ordered_double_buffering<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    let row_count = match lhs.dtype {
        DType::F16 | DType::BF16 => 8,
        _ => 4,
    };
    launch_matmul::<R, E>(
        &Strategy::OrderedDoubleBuffering(Selection::Inferred(OrderedSelectionArgs {
            partition_k: Some(2),
            row_count: Some(row_count),
            rows_per_plane: Some(2),
        })),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn simple_unit_min<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(
        &Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        })),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn simple_unit_max<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(
        &Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MaxTileSize,
        })),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn double_unit<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(&Strategy::DoubleUnit(Default::default()), lhs, rhs, out)
        .map_err(|err| format!("{err:?}"))
}

fn simple_vec_mat<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(
        &Strategy::SimpleVecMat(Selection::Inferred(())),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn double_vec_mat<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(
        &Strategy::DoubleVecMat(Selection::Inferred(())),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn naive<R: CubeRuntime, E: MatmulElement>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, E>(&Strategy::Naive, lhs, rhs, out).map_err(|err| format!("{err:?}"))
}
