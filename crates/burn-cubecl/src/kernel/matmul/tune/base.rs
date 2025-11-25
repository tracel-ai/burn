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
    let output = out.unwrap_or_else(|| init_matmul_output::<R>(&lhs, &rhs, out_dtype));

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
                PRIORITY_HIGH
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
                PRIORITY_HIGH
            } else {
                PRIORITY_MEDIUM
            }
        });

        let unit = TuneGroup::<MatmulAutotuneKey>::new(|key| {
            if !matches!(key.analysis.kind, MatmulKind::General)
                || matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
            {
                PRIORITY_HIGH
            } else {
                PRIORITY_MIN
            }
        });

        let tma = TuneGroup::<MatmulAutotuneKey>::new(|key| {
            // For large matmul, we set the max priority to TMA kernels, higher than any other
            // matmuls, since they are the best kernels no matter what.
            let priority_max = if matches!(key.analysis.kind, MatmulKind::General)
                && matches!(key.analysis.scale_global, MatmulGlobalScale::Large)
            {
                PRIORITY_MAX
            } else {
                PRIORITY_HIGH
            };

            if key.definition.lhs_stride_factor >= 4 && key.definition.rhs_stride_factor >= 4 {
                priority_max
            } else {
                PRIORITY_NEVER
            }
        });

        fn double_buffering_priority(key: &MatmulAutotuneKey, max: i8, min: i8) -> i8 {
            if should_tune_double_buffering(false, key) {
                max
            } else {
                min
            }
        }

        let mut set = TunableSet::new(create_key::<R>, matmul_input_gen::<R>);

        // First entry should always work, since it is considered the fallback.
        set = set.with(
            Tunable::new("matmul_naive", |lhs, rhs, out| {
                launch_matmul::<R>(&Strategy::Naive, lhs, rhs, out)
                    .map_err(|err| std::format!("{err:?}"))
            })
            .group(&unit, |key| {
                if matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
                    || matches!(key.analysis.kind, MatmulKind::InnerProduct)
                {
                    PRIORITY_MAX
                } else {
                    PRIORITY_MIN
                }
            }),
        );

        // Unit VecMat
        for (strategy, double_buf) in [
            (Strategy::SimpleVecMat(Selection::Inferred(())), false),
            (Strategy::DoubleVecMat(Selection::Inferred(())), true),
        ] {
            set = set.with(
                Tunable::new(strategy.to_string(), move |lhs, rhs, out| {
                    launch_matmul::<R>(&strategy, lhs, rhs, out)
                        .map_err(|err| std::format!("{err:?}"))
                })
                .group(&unit, move |key| match double_buf {
                    false => PRIORITY_MAX,
                    true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                }),
            );
        }

        // Unit matmuls
        for tile_size in [
            TileSizeSelection::MaxTileSize,
            TileSizeSelection::MinTileSize,
        ] {
            for (strategy, double_buf) in [
                (
                    Strategy::SimpleUnit(Selection::Inferred(SimpleUnitSelectionArgs {
                        tile_size,
                    })),
                    false,
                ),
                (
                    Strategy::DoubleUnit(Selection::Inferred(
                        cubecl::matmul::kernels::layered::double_unit::DoubleUnitSelectionArgs {
                            tile_size,
                        },
                    )),
                    true,
                ),
            ] {
                set = set.with(
                    Tunable::new(strategy.to_string(), move |lhs, rhs, out| {
                        launch_matmul::<R>(&strategy, lhs, rhs, out)
                            .map_err(|err| format!("{err:?}"))
                    })
                    .group(&unit, move |key| match double_buf {
                        false => PRIORITY_MAX,
                        true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                    }),
                )
            }
        }

        // Accelerated matmuls
        for (tile_kind, tile_group) in [
            (AcceleratedTileKind::Cmma, &cmma),
            (AcceleratedTileKind::Mma, &mma),
        ] {
            for (strategy, double_buf, group_extra) in [
                (
                    Strategy::Simple {
                        read_strategy: ReadingStrategy::Cyclic,
                        selection: Selection::Inferred(SimpleArgs { multi_rows: false }),
                        tile_kind,
                    },
                    false,
                    None,
                ),
                (
                    Strategy::Simple {
                        read_strategy: ReadingStrategy::Cyclic,
                        selection: Selection::Inferred(SimpleArgs { multi_rows: true }),
                        tile_kind,
                    },
                    false,
                    None,
                ),
                (
                    Strategy::OrderedDoubleBuffering {
                        selection: Selection::Inferred(OrderedSelectionArgs {
                            partition_k: Some(2),
                            row_count: Some(4),
                            rows_per_plane: Some(2),
                        }),
                        tile_kind,
                    },
                    true,
                    None,
                ),
                (
                    Strategy::OrderedDoubleBuffering {
                        selection: Selection::Inferred(OrderedSelectionArgs {
                            partition_k: Some(2),
                            row_count: Some(8),
                            rows_per_plane: Some(2),
                        }),
                        tile_kind,
                    },
                    true,
                    None,
                ),
                (
                    Strategy::DoubleBuffering {
                        selection: Selection::Inferred(DoubleBufferingArgs { specialized: false }),
                        tile_kind,
                        read_strategy: PartialReadingStrategy::Tilewise,
                    },
                    true,
                    None,
                ),
                (
                    Strategy::DoubleBuffering {
                        selection: Selection::Inferred(DoubleBufferingArgs { specialized: true }),
                        tile_kind,
                        read_strategy: PartialReadingStrategy::Tilewise,
                    },
                    true,
                    None,
                ),
                (
                    Strategy::Simple {
                        read_strategy: ReadingStrategy::Tma,
                        selection: Selection::Inferred(SimpleArgs { multi_rows: false }),
                        tile_kind,
                    },
                    false,
                    Some(&tma),
                ),
                (
                    Strategy::Simple {
                        read_strategy: ReadingStrategy::Tma,
                        selection: Selection::Inferred(SimpleArgs { multi_rows: true }),
                        tile_kind,
                    },
                    false,
                    Some(&tma),
                ),
                (
                    Strategy::Specialized {
                        selection: Selection::Inferred(()),
                        tile_kind,
                    },
                    true,
                    Some(&tma),
                ),
            ] {
                let priority_within_group =
                    |key: &MatmulAutotuneKey, double_buf: bool| match double_buf {
                        false => PRIORITY_MAX,
                        true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                    };
                let mut tunable = Tunable::new(strategy.to_string(), move |lhs, rhs, out| {
                    launch_matmul::<R>(&strategy, lhs, rhs, out).map_err(|err| format!("{err:?}"))
                });

                // tile group
                tunable = tunable.group(tile_group, move |key| {
                    priority_within_group(key, double_buf)
                });

                // extra group
                if let Some(group) = group_extra {
                    tunable =
                        tunable.group(group, move |key| priority_within_group(key, double_buf));
                }
                set = set.with(tunable);
            }
        }

        set
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
