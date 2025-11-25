use burn_tensor::DType;
use cubecl::{
    matmul::{
        AcceleratedTileKind, PartialReadingStrategy, ReadingStrategy, Strategy,
        components::MatmulKind,
        kernels::layered::{
            Selection, TileSizeSelection, double_buffering::DoubleBufferingArgs,
            ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
            simple_unit::SimpleUnitSelectionArgs,
        },
        tune_key::{
            MatmulAutotuneKey, MatmulElemType, MatmulGlobalScale, should_tune_double_buffering,
        },
    },
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};

use crate::{
    CubeRuntime, CubeTuneId,
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
pub fn matmul_autotune<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    out_dtype: DType,
) -> CubeTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));

    let client = lhs.client.clone();

    static TUNER: LocalTuner<MatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: i8 = 3;
        const PRIORITY_HIGH: i8 = 2;
        const PRIORITY_MEDIUM: i8 = 1;
        const PRIORITY_MIN: i8 = 0;
        const PRIORITY_NEVER: i8 = -1;

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

        let mma = TuneGroup::<MatmulAutotuneKey>::new(|key| {
            if matches!(
                key.analysis.kind,
                // General is usually bad, but I think shapes like 16x8196 would be classed as
                // general and are very good with MMA
                // Should highly degenerated matrices that aren't VecMat have their own class?
                MatmulKind::General | MatmulKind::VecMat | MatmulKind::MatVec
            ) {
                PRIORITY_MAX
            } else {
                PRIORITY_MEDIUM
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

        fn double_buffering_priority(key: &MatmulAutotuneKey, max: i8, min: i8) -> i8 {
            if should_tune_double_buffering(false, key) {
                max
            } else {
                min
            }
        }

        fn tma_priority(key: &MatmulAutotuneKey) -> i8 {
            if key.definition.lhs_stride_factor >= 4 && key.definition.rhs_stride_factor >= 4 {
                PRIORITY_MAX
            } else {
                PRIORITY_NEVER
            }
        }

        type Input<R> = (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>);

        fn accelerated<R: CubeRuntime, const MMA: bool>(
            set: TunableSet<MatmulAutotuneKey, Input<R>, ()>,
            tile: &TuneGroup<MatmulAutotuneKey>,
        ) -> TunableSet<MatmulAutotuneKey, Input<R>, ()> {
            let odd = TuneGroup::<MatmulAutotuneKey>::new(|key| {
                if key.definition.lhs_pow2_factor == 0 || key.definition.rhs_pow2_factor == 0 {
                    PRIORITY_MAX
                } else {
                    PRIORITY_MIN
                }
            });

            set.with(Tunable::new(matmul_simple::<R, MMA>).group(tile, |_| PRIORITY_MAX))
                .with(
                    Tunable::new(matmul_simple_tma::<R, MMA>)
                        .group(tile, tma_priority)
                        .group(&odd, tma_priority),
                )
                .with(
                    Tunable::new(matmul_simple_multi_rows::<R, MMA>).group(tile, |_| PRIORITY_MAX),
                )
                .with(
                    // Ordered should be tried most of the time.
                    Tunable::new(matmul_ordered_double_buffering::<R, MMA>)
                        .group(tile, |_| PRIORITY_MAX),
                )
                .with(
                    Tunable::new(matmul_double_buffering_specialized::<R, MMA>)
                        .group(tile, |key| {
                            double_buffering_priority(key, PRIORITY_HIGH, PRIORITY_MEDIUM)
                        })
                        .group(&odd, |_| PRIORITY_MAX),
                )
                .with(
                    Tunable::new(matmul_double_buffering::<R, MMA>)
                        .group(tile, |key| {
                            double_buffering_priority(key, PRIORITY_HIGH, PRIORITY_MEDIUM)
                        })
                        .group(&odd, |_| PRIORITY_MAX),
                )
                .with(
                    Tunable::new(matmul_double_buffering_tma::<R, MMA>)
                        // TMA is often the best double buffering algorithm when available
                        .group(tile, |key| {
                            double_buffering_priority(key, PRIORITY_MAX, PRIORITY_MEDIUM)
                                .min(tma_priority(key))
                        })
                        .group(&odd, tma_priority),
                )
                .with(
                    Tunable::new(matmul_specialized_tma::<R, MMA>)
                        // TMA is often the best double buffering algorithm when available
                        .group(tile, |key| {
                            double_buffering_priority(key, PRIORITY_MAX, PRIORITY_MEDIUM)
                                .min(tma_priority(key))
                        })
                        .group(&odd, tma_priority),
                )
        }

        let set = TunableSet::new(create_key, matmul_input_gen)
            .with(Tunable::new(naive).group(&unit, |key| {
                if matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
                    || matches!(key.analysis.kind, MatmulKind::InnerProduct)
                {
                    PRIORITY_MAX
                } else {
                    PRIORITY_MIN
                }
            }))
            .with(
                Tunable::new(|lhs, rhs, out| {
                    simple_unit(lhs, rhs, out, TileSizeSelection::MinTileSize)
                })
                .group(&unit, |key| {
                    if matches!(key.analysis.kind, MatmulKind::General)
                        && matches!(key.analysis.scale_global, MatmulGlobalScale::Large)
                    {
                        PRIORITY_MAX
                    } else {
                        PRIORITY_HIGH
                    }
                }),
            )
            .with(
                Tunable::new(|lhs, rhs, out| {
                    simple_unit(lhs, rhs, out, TileSizeSelection::MaxTileSize)
                })
                .group(&unit, |_| PRIORITY_MAX),
            )
            .with(Tunable::new(simple_vec_mat).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(double_vec_mat).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(double_unit).group(&unit, |key| {
                double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH)
            }));

        let set = accelerated::<R, false>(set, &cmma);
        accelerated::<R, true>(set, &mma)
    });

    TUNER.execute(
        &CubeTuneId::new(&lhs.client, &lhs.device),
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
    MatmulAutotuneKey::generate(
        &lhs.client,
        &lhs.shape.dims,
        &rhs.shape.dims,
        &lhs.strides,
        &rhs.strides,
        MatmulElemType {
            elem: lhs.dtype.into(),
            quantized: matches!(lhs.dtype, DType::QFloat(_)),
        },
        MatmulElemType {
            elem: rhs.dtype.into(),
            quantized: matches!(rhs.dtype, DType::QFloat(_)),
        },
        MatmulElemType {
            elem: out.dtype.into(),
            quantized: matches!(out.dtype, DType::QFloat(_)),
        },
    )
}

fn tile_kind(mma: bool) -> AcceleratedTileKind {
    if mma {
        AcceleratedTileKind::Mma
    } else {
        AcceleratedTileKind::Cmma
    }
}

fn matmul_simple<R: CubeRuntime, const MMA: bool>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul(
        &Strategy::Simple {
            read_strategy: ReadingStrategy::Cyclic,
            selection: Selection::Inferred(SimpleArgs { multi_rows: false }),
            tile_kind: tile_kind(MMA),
        },
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_simple_tma<R: CubeRuntime, const MMA: bool>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        return Err("TMA can't be used for quantization right now".into());
    }
    launch_matmul(
        &Strategy::Simple {
            read_strategy: ReadingStrategy::Tma,
            selection: Selection::Inferred(SimpleArgs { multi_rows: false }),
            tile_kind: tile_kind(MMA),
        },
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_simple_multi_rows<R: CubeRuntime, const MMA: bool>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul(
        &Strategy::Simple {
            read_strategy: ReadingStrategy::Cyclic,
            selection: Selection::Inferred(SimpleArgs { multi_rows: true }),
            tile_kind: tile_kind(MMA),
        },
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_double_buffering<R: CubeRuntime, const MMA: bool>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul(
        &Strategy::DoubleBuffering {
            read_strategy: PartialReadingStrategy::Tilewise,
            selection: Selection::Inferred(DoubleBufferingArgs { specialized: false }),
            tile_kind: tile_kind(MMA),
        },
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_double_buffering_tma<R: CubeRuntime, const MMA: bool>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        return Err("TMA can't be used for quantization right now".into());
    }
    launch_matmul(
        &Strategy::DoubleBuffering {
            read_strategy: PartialReadingStrategy::Tma,
            selection: Selection::Inferred(DoubleBufferingArgs { specialized: false }),
            tile_kind: tile_kind(MMA),
        },
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_double_buffering_specialized<R: CubeRuntime, const MMA: bool>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul(
        &Strategy::DoubleBuffering {
            read_strategy: PartialReadingStrategy::Tilewise,
            selection: Selection::Inferred(DoubleBufferingArgs { specialized: true }),
            tile_kind: tile_kind(MMA),
        },
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_specialized_tma<R: CubeRuntime, const MMA: bool>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        return Err("TMA can't be used for quantization right now".into());
    }
    launch_matmul(
        &Strategy::Specialized {
            selection: Selection::Inferred(()),
            tile_kind: tile_kind(MMA),
        },
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn matmul_ordered_double_buffering<R: CubeRuntime, const MMA: bool>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    let row_count = match lhs.dtype {
        DType::F16 | DType::BF16 => 8,
        _ => 4,
    };
    launch_matmul(
        &Strategy::OrderedDoubleBuffering {
            selection: Selection::Inferred(OrderedSelectionArgs {
                partition_k: Some(2),
                row_count: Some(row_count),
                rows_per_plane: Some(2),
            }),
            tile_kind: tile_kind(MMA),
        },
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn simple_unit<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
    tile_size: TileSizeSelection,
) -> Result<(), String> {
    launch_matmul(
        &Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs { tile_size })),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn double_unit<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul(&Strategy::DoubleUnit(Default::default()), lhs, rhs, out)
        .map_err(|err| format!("{err:?}"))
}

fn simple_vec_mat<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul(
        &Strategy::SimpleVecMat(Selection::Inferred(())),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn double_vec_mat<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul(
        &Strategy::DoubleVecMat(Selection::Inferred(())),
        lhs,
        rhs,
        out,
    )
    .map_err(|err| format!("{err:?}"))
}

fn naive<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: CubeTensor<R>,
) -> Result<(), String> {
    launch_matmul(&Strategy::Naive, lhs, rhs, out).map_err(|err| format!("{err:?}"))
}
