use burn_tensor::DType;
use cubecl::{
    matmul::{
        AsyncReadingStrategy, Strategy, SyncPartialReadingStrategy, SyncReadingStrategy,
        components::{AccG, MatmulKind, MatmulPrecision, MatrixPrecision},
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
    CubeElement, CubeRuntime, CubeTuneId,
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
pub fn matmul_autotune<
    R: CubeRuntime,
    MP: MatmulPrecision<Acc: MatrixPrecision<Global: CubeElement>>,
>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output::<R, AccG<MP>>(&lhs, &rhs));

    let client = lhs.client.clone();

    static TUNER: LocalTuner<MatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: u8 = 3;
        const PRIORITY_HIGH: u8 = 2;
        const PRIORITY_MEDIUM: u8 = 1;
        const PRIORITY_MIN: u8 = 0;

        let cmma = TuneGroup::<MatmulAutotuneKey>::new(|key| {
            if matches!(
                key.analysis.kind,
                MatmulKind::General
                // Those variants are just because the unit alternatives aren't very good yet.
                | MatmulKind::VecMat | MatmulKind::MatVec
            ) {
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
            .with(Tunable::new(naive::<R, MP>).group(&unit, |key| {
                if matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
                    || matches!(key.analysis.kind, MatmulKind::InnerProduct)
                {
                    PRIORITY_MAX
                } else {
                    PRIORITY_MIN
                }
            }))
            .with(Tunable::new(simple_unit_min::<R, MP>).group(&unit, |key| {
                if matches!(key.analysis.kind, MatmulKind::General)
                    && matches!(key.analysis.scale_global, MatmulGlobalScale::Large)
                {
                    PRIORITY_MAX
                } else {
                    PRIORITY_HIGH
                }
            }))
            .with(Tunable::new(simple_unit_max::<R, MP>).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(simple_vec_mat::<R, MP>).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(double_vec_mat::<R, MP>).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(double_unit::<R, MP>).group(&unit, |key| {
                double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH)
            }))
            .with(Tunable::new(matmul_simple::<R, MP>).group(&cmma, |_| PRIORITY_MAX))
            .with(Tunable::new(matmul_simple_tma::<R, MP>).group(&cmma, |_| PRIORITY_MAX))
            .with(Tunable::new(matmul_simple_multi_rows::<R, MP>).group(&cmma, |_| PRIORITY_MAX))
            .with(
                // Ordered should be tried most of the time.
                Tunable::new(matmul_ordered_double_buffering::<R, MP>)
                    .group(&cmma, |_| PRIORITY_MAX),
            )
            .with(
                Tunable::new(matmul_double_buffering_specialized::<R, MP>)
                    .group(&cmma, |key| {
                        double_buffering_priority(key, PRIORITY_HIGH, PRIORITY_MEDIUM)
                    })
                    .group(&odd, |_| PRIORITY_MAX),
            )
            .with(
                Tunable::new(matmul_double_buffering::<R, MP>)
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

fn matmul_simple<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(
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

fn matmul_simple_multi_rows<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(
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

fn matmul_double_buffering<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(
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

fn matmul_double_buffering_specialized<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(
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

fn matmul_ordered_double_buffering<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    let row_count = match lhs.dtype {
        DType::F16 | DType::BF16 => 8,
        _ => 4,
    };
    launch_matmul::<R, MP>(
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

fn simple_unit_min<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(
        &Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MinTileSize,
        })),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn simple_unit_max<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(
        &Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
            tile_size: TileSizeSelection::MaxTileSize,
        })),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn double_unit<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(&Strategy::DoubleUnit(Default::default()), lhs, rhs, out)
        .map_err(|err| format!("{err:?}"))
}

fn simple_vec_mat<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(
        &Strategy::SimpleVecMat(Selection::Inferred(())),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_simple_tma<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(
        &Strategy::SimpleBarrier(AsyncReadingStrategy::Tma),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn double_vec_mat<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul::<R, MP>(
        &Strategy::DoubleVecMat(Selection::Inferred(())),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn naive<R: CubeRuntime, MP: MatmulPrecision>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    if matches!(lhs.dtype, DType::QFloat(_)) || matches!(rhs.dtype, DType::QFloat(_)) {
        return Err("QFloat isn't stable yet with quantized inputs".into());
    }

    launch_matmul::<R, MP>(&Strategy::Naive, lhs, rhs, out).map_err(|err| format!("{err:?}"))
}
