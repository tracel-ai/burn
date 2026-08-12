use crate::{
    CubeRuntime, CubeTuneId,
    kernel::matmul::{
        launch_matmul, launch_matmul_naive, tune::bounds::with_matmul_bounds,
        utils::init_matmul_output,
    },
    tensor::CubeTensor,
};
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    client::ComputeClient,
    std::tensor::MatrixBatchLayout,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};
use cubek::matmul::{
    components::tile::TileMatmulKind,
    definition::{MatmulElems, MatmulGlobalElems, MatmulKind, adjust_dtypes},
    routines::{
        BlueprintStrategy, TileSizeSelection,
        batch::{
            double_buffering::DoubleBufferingArgs, double_unit::DoubleUnitSelectionArgs,
            ordered_double_buffering::OrderedSelectionArgs, simple::SimpleArgs,
            simple_unit::SimpleUnitSelectionArgs,
        },
        cpu_gemm::CpuGemmStrategy,
        gemm::GemmStrategy,
    },
    strategy::{
        MatmulAutotuneKey, MatmulGlobalScale, MatmulProblemDefinition, Strategy,
        should_tune_double_buffering,
    },
};

pub(super) type Inputs<R> = (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>);

/// Whether the device can run `tile_matmul` with the element types of the matmul `definition`.
///
/// The kernel runs on the *register* types, not the global ones: the selection paths promote
/// them with [`adjust_dtypes`] when the tile matmul needs an accelerator (f32 to tf32, flex32 to
/// f16). Without the same promotion here, every f32 problem would look unsupported and the tf32
/// tensor core path would be lost.
pub(crate) fn tile_matmul_supported<R: CubeRuntime>(
    client: &ComputeClient<R>,
    tile_matmul: TileMatmulKind,
    definition: &MatmulProblemDefinition,
) -> bool {
    let mut elems = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: definition.elem_lhs,
        rhs: definition.elem_rhs,
        out: definition.elem_out,
    });
    adjust_dtypes(client, &mut elems, tile_matmul.requires_accelerator());

    !tile_matmul
        .supported_sizes(
            client,
            elems.lhs_register,
            elems.rhs_register,
            elems.acc_register,
        )
        .is_empty()
}

fn matmul_input_gen<R: CubeRuntime>(
    _key: &MatmulAutotuneKey,
    (lhs, rhs, out): &Inputs<R>,
) -> Inputs<R> {
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

    // Short-circuit: zero-sized matmul produces a zero-sized output.
    if lhs.meta.shape().iter().any(|&d| d == 0) || rhs.meta.shape().iter().any(|&d| d == 0) {
        return output;
    }

    let client = lhs.client.clone();
    let tune_client = client.clone();
    let num_cpu_cores = client.properties().hardware.num_cpu_cores;

    static TUNER: LocalTuner<MatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(move || {
        const PRIORITY_MAX: i8 = 3;
        const PRIORITY_HIGH: i8 = 2;
        const PRIORITY_MEDIUM: i8 = 1;
        const PRIORITY_MIN: i8 = 0;
        const PRIORITY_NEVER: i8 = -1;

        let accelerated = TuneGroup::<MatmulAutotuneKey>::new("accelerated", |key| {
            if matches!(key.analysis.kind, MatmulKind::General) {
                match key.analysis.scale_global {
                    MatmulGlobalScale::Large => PRIORITY_MAX,
                    _ => PRIORITY_HIGH,
                }

            // In some case when a relayout can be fused (no call to into_contiguous) it's better
            // to use accelerated matmul.
            //
            // TODO: Actually implement good gemv with fused relayout.
            } else if matches!(key.analysis.kind, MatmulKind::MatVec | MatmulKind::VecMat) {
                PRIORITY_MAX
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
                PRIORITY_MEDIUM
            }
        });

        let tma = TuneGroup::<MatmulAutotuneKey>::new("tma", |key| {
            // Zero-sized matmuls cannot use TMA kernels.
            if key.definition.m == 0 || key.definition.n == 0 || key.definition.k == 0 {
                return PRIORITY_NEVER;
            }

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

        let gemv = TuneGroup::<MatmulAutotuneKey>::new("gemv", move |key| {
            if num_cpu_cores.is_some() {
                return PRIORITY_MAX;
            }

            if matches!(key.analysis.kind, MatmulKind::MatVec) {
                // LHS is the matrix
                match key.definition.matrix_layout_lhs {
                    MatrixBatchLayout::Contiguous => PRIORITY_MAX,
                    MatrixBatchLayout::MildlyPermuted { transposed, .. } => {
                        // We don't yet have algo which are good for col major matvec.
                        if transposed {
                            PRIORITY_HIGH
                        } else {
                            PRIORITY_MAX
                        }
                    }
                    // Every algo will need to relayout, in this case, we should take the optimal
                    // kernel with a gemv.
                    MatrixBatchLayout::HighlyPermuted => PRIORITY_MAX,
                }
            } else if matches!(key.analysis.kind, MatmulKind::VecMat) {
                // RHS is the matrix
                match key.definition.matrix_layout_rhs {
                    // We don't have good algos for row major vecmat.
                    MatrixBatchLayout::Contiguous => PRIORITY_HIGH,
                    MatrixBatchLayout::MildlyPermuted { transposed, .. } => {
                        // Best algo is col major vec mat.
                        if transposed {
                            PRIORITY_MAX
                        } else {
                            PRIORITY_HIGH
                        }
                    }
                    // TODO: Actually do the correct relayout here.
                    //
                    // Every algo will need to relayout, in this case, we should take the optimal
                    // kernel with a gemv.
                    MatrixBatchLayout::HighlyPermuted => PRIORITY_HIGH,
                }
            } else {
                PRIORITY_NEVER
            }
        });

        // CPU-only group
        let cpu =
            TuneGroup::<MatmulAutotuneKey>::new("cpu", move |_key| match num_cpu_cores.is_some() {
                true => PRIORITY_MAX,
                false => PRIORITY_NEVER,
            });

        fn double_buffering_priority(key: &MatmulAutotuneKey, max: i8, min: i8) -> i8 {
            if should_tune_double_buffering(false, key) {
                max
            } else {
                min
            }
        }

        let mut set = TunableSet::new(create_key::<R>, matmul_input_gen::<R>);

        set = with_matmul_bounds(set);

        // First entry should always work, since it is considered the fallback.
        set = set.with(
            Tunable::new("matmul_naive", |(lhs, rhs, out)| {
                launch_matmul_naive::<R>(&Strategy::Naive, lhs, rhs, out)
                    .map_err(|err| std::format!("{err:?}"))
            })
            .group(&unit, |key| {
                if matches!(key.analysis.kind, MatmulKind::InnerProduct) {
                    PRIORITY_MAX
                } else if matches!(key.analysis.scale_global, MatmulGlobalScale::Small) {
                    PRIORITY_HIGH
                } else {
                    PRIORITY_MIN
                }
            }),
        );

        // Matrix Vector multiplication kernels.
        for (strategy, double_buf) in [
            (
                Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into())),
                true,
            ),
            (
                Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
                false,
            ),
            (
                Strategy::Gemm(BlueprintStrategy::Inferred(Default::default())),
                false,
            ),
            (
                Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(Default::default())),
                false,
            ),
        ] {
            set = set.with(
                Tunable::new(&strategy.to_string(), move |(lhs, rhs, out)| {
                    launch_matmul::<R>(&strategy, lhs, rhs, out)
                        .map_err(|err| std::format!("{err:?}"))
                })
                .group(&gemv, move |key| match double_buf {
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
                    Tunable::new(&strategy.to_string(), move |(lhs, rhs, out)| {
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

        // Gemm no stage
        // In unit because not accelerated
        let gemm_no_stage_strategy = Strategy::Gemm(BlueprintStrategy::Inferred(GemmStrategy {
            target_num_planes: None,
        }));
        set = set.with(
            Tunable::new(
                &gemm_no_stage_strategy.to_string(),
                move |(lhs, rhs, out)| {
                    launch_matmul::<R>(&gemm_no_stage_strategy, lhs, rhs, out)
                        .map_err(|err| format!("{err:?}"))
                },
            )
            .group(&unit, move |_key| PRIORITY_MAX),
        );

        // CPU GEMM (CPU-only via the `cpu` group; the size limit is specific to this strategy).
        let cpu_gemm_strategy =
            Strategy::CpuGemm(BlueprintStrategy::Inferred(CpuGemmStrategy::default()));
        set = set.with(
            Tunable::new(&cpu_gemm_strategy.to_string(), move |(lhs, rhs, out)| {
                launch_matmul::<R>(&cpu_gemm_strategy, lhs, rhs, out)
                    .map_err(|err| format!("{err:?}"))
            })
            .group(&cpu, move |_key| PRIORITY_MAX),
        );

        // Accelerated matmuls
        for (strategy, double_buf, group_extra, tile_group, tile_matmul) in [
            (
                Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                false,
                None,
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                false,
                None,
                &accelerated,
                TileMatmulKind::Mma,
            ),
            (
                Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                false,
                None,
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                false,
                None,
                &accelerated,
                TileMatmulKind::Mma,
            ),
            (
                Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(4),
                    rows_per_plane: Some(2),
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                true,
                None,
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(4),
                    rows_per_plane: Some(2),
                    tile_matmul: TileMatmulKind::Mma,
                })),
                true,
                None,
                &accelerated,
                TileMatmulKind::Mma,
            ),
            (
                Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(8),
                    rows_per_plane: Some(2),
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                true,
                None,
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(8),
                    rows_per_plane: Some(2),
                    tile_matmul: TileMatmulKind::Mma,
                })),
                true,
                None,
                &accelerated,
                TileMatmulKind::Mma,
            ),
            (
                Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: false,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                true,
                None,
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: false,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                true,
                None,
                &accelerated,
                TileMatmulKind::Mma,
            ),
            (
                Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: true,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                true,
                None,
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: true,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                true,
                None,
                &accelerated,
                TileMatmulKind::Mma,
            ),
            (
                Strategy::SpecializedCyclicCmma(BlueprintStrategy::Inferred(().into())),
                true,
                None,
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into())),
                true,
                None,
                &accelerated,
                TileMatmulKind::Mma,
            ),
            (
                Strategy::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                false,
                Some(&tma),
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                false,
                Some(&tma),
                &accelerated,
                TileMatmulKind::Mma,
            ),
            (
                Strategy::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                false,
                Some(&tma),
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                false,
                Some(&tma),
                &accelerated,
                TileMatmulKind::Mma,
            ),
            (
                Strategy::SpecializedTmaCmma(BlueprintStrategy::Inferred(().into())),
                true,
                Some(&tma),
                &accelerated,
                TileMatmulKind::Cmma,
            ),
            (
                Strategy::SpecializedTmaMma(BlueprintStrategy::Inferred(().into())),
                true,
                Some(&tma),
                &accelerated,
                TileMatmulKind::Mma,
            ),
        ] {
            let mut tunable = Tunable::new(&strategy.to_string(), move |(lhs, rhs, out)| {
                launch_matmul::<R>(&strategy, lhs, rhs, out).map_err(|err| format!("{err:?}"))
            });

            // Accelerated kernels are demoted when the device doesn't support the tile matmul
            // they are built on, otherwise they would be compiled just to fail. They keep the
            // minimum priority rather than being discarded, so they remain a last resort and
            // the tune plan can never end up empty.
            let accelerated_priority = move |key: &MatmulAutotuneKey, client: &ComputeClient<R>| {
                if !tile_matmul_supported::<R>(client, tile_matmul, &key.definition) {
                    return PRIORITY_MIN;
                }

                match double_buf {
                    false => PRIORITY_MAX,
                    true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                }
            };

            // tile group
            let client_tile = tune_client.clone();
            tunable = tunable.group(tile_group, move |key| {
                accelerated_priority(key, &client_tile)
            });

            // extra group
            if let Some(group) = group_extra {
                let client_extra = tune_client.clone();
                tunable = tunable.group(group, move |key| accelerated_priority(key, &client_extra));
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

fn create_key<R: CubeRuntime>((lhs, rhs, out): &Inputs<R>) -> MatmulAutotuneKey {
    MatmulAutotuneKey::generate(
        &lhs.client,
        lhs.meta.shape(),
        rhs.meta.shape(),
        lhs.meta.strides(),
        rhs.meta.strides(),
        dtype_to_storage_type(lhs.dtype),
        dtype_to_storage_type(rhs.dtype),
        dtype_to_storage_type(out.dtype),
        lhs.try_scheme(),
        rhs.try_scheme(),
    )
}
