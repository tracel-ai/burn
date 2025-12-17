use crate::{
    CubeRuntime, CubeTuneId,
    kernel::matmul::{launch_matmul, utils::init_matmul_output},
    tensor::CubeTensor,
};
use burn_backend::DType;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner};
use cubek::matmul::{
    definition::{MatmulElemType, MatmulKind},
    launch::{MatmulAutotuneKey, MatmulGlobalScale, Strategy, should_tune_double_buffering},
    routines::{
        BlueprintStrategy, TileSizeSelection, double_buffering::DoubleBufferingArgs,
        double_unit::DoubleUnitSelectionArgs, ordered_double_buffering::OrderedSelectionArgs,
        simple::SimpleArgs, simple_unit::SimpleUnitSelectionArgs,
    },
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

        let cmma = TuneGroup::<MatmulAutotuneKey>::new("cmma", |key| {
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

        let mma = TuneGroup::<MatmulAutotuneKey>::new("mma", |key| {
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

        let unit = TuneGroup::<MatmulAutotuneKey>::new("unit", |key| {
            if !matches!(key.analysis.kind, MatmulKind::General)
                || matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
            {
                PRIORITY_HIGH
            } else {
                PRIORITY_MIN
            }
        });

        let tma = TuneGroup::<MatmulAutotuneKey>::new("tma", |key| {
            // For large matmul, we set the max priority to TMA kernels, higher than any other
            // matmuls, since they are the best kernels no matter what.
            //
            // But only when all axis are large.
            let max_axis = usize::max(key.definition.m, key.definition.n);
            let max_axis = usize::max(key.definition.k, max_axis);

            let min_axis = usize::min(key.definition.m, key.definition.n);
            let min_axis = usize::min(key.definition.k, min_axis);

            let skewed_factor = max_axis / min_axis;

            let priority_max = if matches!(key.analysis.kind, MatmulKind::General)
                && matches!(key.analysis.scale_global, MatmulGlobalScale::Large)
                && skewed_factor < 4
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
            (
                Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
                false,
            ),
            (
                Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into())),
                true,
            ),
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
                    Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                        tile_size,
                    })),
                    false,
                ),
                (
                    Strategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                        tile_size,
                    })),
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
        for (strategy, double_buf, group_extra, tile_group) in [
            (
                Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                })),
                false,
                None,
                &cmma,
            ),
            (
                Strategy::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                })),
                false,
                None,
                &mma,
            ),
            (
                Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                })),
                false,
                None,
                &cmma,
            ),
            (
                Strategy::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                })),
                false,
                None,
                &mma,
            ),
            (
                Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(4),
                    rows_per_plane: Some(2),
                })),
                true,
                None,
                &cmma,
            ),
            (
                Strategy::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(4),
                    rows_per_plane: Some(2),
                })),
                true,
                None,
                &mma,
            ),
            (
                Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(8),
                    rows_per_plane: Some(2),
                })),
                true,
                None,
                &cmma,
            ),
            (
                Strategy::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(8),
                    rows_per_plane: Some(2),
                })),
                true,
                None,
                &mma,
            ),
            (
                Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: false,
                })),
                true,
                None,
                &cmma,
            ),
            (
                Strategy::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: false,
                })),
                true,
                None,
                &mma,
            ),
            (
                Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: true,
                })),
                true,
                None,
                &cmma,
            ),
            (
                Strategy::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: true,
                })),
                true,
                None,
                &mma,
            ),
            (
                Strategy::SpecializedCyclicCmma(BlueprintStrategy::Inferred(().into())),
                true,
                None,
                &cmma,
            ),
            (
                Strategy::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into())),
                true,
                None,
                &mma,
            ),
            (
                Strategy::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                })),
                false,
                Some(&tma),
                &cmma,
            ),
            (
                Strategy::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                })),
                false,
                Some(&tma),
                &mma,
            ),
            (
                Strategy::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                })),
                false,
                Some(&tma),
                &cmma,
            ),
            (
                Strategy::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                })),
                false,
                Some(&tma),
                &mma,
            ),
            (
                Strategy::SpecializedTmaCmma(BlueprintStrategy::Inferred(().into())),
                true,
                Some(&tma),
                &cmma,
            ),
            (
                Strategy::SpecializedTmaMma(BlueprintStrategy::Inferred(().into())),
                true,
                Some(&tma),
                &mma,
            ),
        ] {
            let priority_within_group = |key: &MatmulAutotuneKey, double_buf: bool| match double_buf
            {
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
                tunable = tunable.group(group, move |key| priority_within_group(key, double_buf));
            }
            set = set.with(tunable);
        }

        set
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
            dtype: lhs.dtype.into(),
            quantized: matches!(lhs.dtype, DType::QFloat(_)),
        },
        MatmulElemType {
            dtype: rhs.dtype.into(),
            quantized: matches!(rhs.dtype, DType::QFloat(_)),
        },
        MatmulElemType {
            dtype: out.dtype.into(),
            quantized: matches!(out.dtype, DType::QFloat(_)),
        },
    )
}
