use super::optimization::MatmulOptimizationTuneArg;
use crate::{
    CubeFusionHandle,
    engine::trace::TuneOutput,
    optim::matmul::{AcceleratedTileKind, FusedMatmulSelector},
    tune::{TuneContext, TuneInput},
};
use burn_fusion::stream::Context;
use cubecl::{
    AutotuneKey, CubeTuneId, Runtime,
    std::tensor::MatrixBatchLayout,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};
use cubek::matmul::{
    definition::MatmulKind,
    launch::{MatmulAutotuneKey, MatmulGlobalScale, should_tune_double_buffering},
};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct FusedMatmulAutotuneKey {
    matmul_key: MatmulAutotuneKey,
    #[autotune(anchor)]
    num_out_buffers: usize,
    #[autotune(anchor)]
    num_ops: usize,
}

/// Executes autotune on matmul operations
pub fn fused_matmul_autotune<R: Runtime>(
    optimization: MatmulOptimizationTuneArg<R>,
    context: &mut Context<CubeFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedMatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: i8 = 3;
        const PRIORITY_HIGH: i8 = 2;
        const PRIORITY_MEDIUM: i8 = 1;
        const PRIORITY_MIN: i8 = 0;
        const PRIORITY_NEVER: i8 = -1;

        let accelerated = TuneGroup::<FusedMatmulAutotuneKey>::new("accelerated", |key| {
            if matches!(key.matmul_key.analysis.kind, MatmulKind::General) {
                match key.matmul_key.analysis.scale_global {
                    MatmulGlobalScale::Large => PRIORITY_MAX,
                    _ => PRIORITY_HIGH,
                }
            } else if matches!(
                key.matmul_key.analysis.kind,
                MatmulKind::MatVec | MatmulKind::VecMat
            ) {
                PRIORITY_HIGH
            } else {
                PRIORITY_MEDIUM
            }
        });

        let unit = TuneGroup::<FusedMatmulAutotuneKey>::new("unit", |key| {
            if !matches!(key.matmul_key.analysis.kind, MatmulKind::General)
                || matches!(
                    key.matmul_key.analysis.scale_global,
                    MatmulGlobalScale::Small
                )
            {
                PRIORITY_HIGH
            } else {
                PRIORITY_MEDIUM
            }
        });

        let gemv = TuneGroup::<FusedMatmulAutotuneKey>::new("gemv", |key| {
            if matches!(key.matmul_key.analysis.kind, MatmulKind::MatVec) {
                // LHS is the matrix.
                match key.matmul_key.definition.matrix_layout_lhs {
                    MatrixBatchLayout::Contiguous => PRIORITY_MAX,
                    MatrixBatchLayout::MildlyPermuted { transposed, .. } => {
                        if transposed {
                            PRIORITY_HIGH
                        } else {
                            PRIORITY_MAX
                        }
                    }
                    MatrixBatchLayout::HighlyPermuted => PRIORITY_MAX,
                }
            } else if matches!(key.matmul_key.analysis.kind, MatmulKind::VecMat) {
                // RHS is the matrix.
                match key.matmul_key.definition.matrix_layout_rhs {
                    MatrixBatchLayout::Contiguous => PRIORITY_HIGH,
                    MatrixBatchLayout::MildlyPermuted { transposed, .. } => {
                        if transposed {
                            PRIORITY_MAX
                        } else {
                            PRIORITY_HIGH
                        }
                    }
                    MatrixBatchLayout::HighlyPermuted => PRIORITY_HIGH,
                }
            } else {
                PRIORITY_NEVER
            }
        });

        let odd = TuneGroup::<FusedMatmulAutotuneKey>::new("odd", |key| {
            if key.matmul_key.definition.lhs_pow2_factor == 0
                || key.matmul_key.definition.rhs_pow2_factor == 0
            {
                PRIORITY_MAX
            } else {
                PRIORITY_MIN
            }
        });

        fn double_buffering_priority(key: &FusedMatmulAutotuneKey, max: i8, min: i8) -> i8 {
            if should_tune_double_buffering(key.num_out_buffers > 1, &key.matmul_key) {
                max
            } else {
                min
            }
        }

        // First entry should always work, since it is considered the fallback.
        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>).with(
            Tunable::new("fused_matmul_fallback", tune_fallback::<R>).group(&unit, |key| {
                if matches!(key.matmul_key.analysis.kind, MatmulKind::InnerProduct) {
                    PRIORITY_MAX
                } else if matches!(
                    key.matmul_key.analysis.scale_global,
                    MatmulGlobalScale::Small
                ) {
                    PRIORITY_HIGH
                } else {
                    PRIORITY_MIN
                }
            }),
        );

        // Vector-matrix kernels.
        for (selector, double_buf) in [
            (FusedMatmulSelector::SimpleVecMat, false),
            (FusedMatmulSelector::DoubleVecMat, true),
            (FusedMatmulSelector::GemvPlaneParallel, false),
            (FusedMatmulSelector::GemvUnitPerpendicular, false),
        ] {
            set = set.with(
                Tunable::new(selector.name(), move |input| {
                    tune_fused::<R>(input, selector)
                })
                .group(&gemv, move |key| match double_buf {
                    false => PRIORITY_MAX,
                    true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                }),
            );
        }

        // Unit matmuls
        for (selector, double_buf) in [
            (FusedMatmulSelector::SimpleUnit, false),
            (FusedMatmulSelector::DoubleUnit, true),
        ] {
            set = set.with(
                Tunable::new(selector.name(), move |input| {
                    tune_fused::<R>(input, selector)
                })
                .group(&unit, move |key| match double_buf {
                    false => PRIORITY_MAX,
                    true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                }),
            );
        }

        // Accelerated matmuls
        for tile_matmul in [AcceleratedTileKind::Cmma, AcceleratedTileKind::Mma] {
            for (selector, double_buf, extra_group) in [
                (
                    FusedMatmulSelector::Simple {
                        multi_rows: false,
                        tile_matmul,
                    },
                    false,
                    None,
                ),
                (
                    FusedMatmulSelector::Simple {
                        multi_rows: true,
                        tile_matmul,
                    },
                    false,
                    None,
                ),
                (
                    FusedMatmulSelector::OrderedDoubleBuffering { tile_matmul },
                    true,
                    None,
                ),
                (
                    FusedMatmulSelector::DoubleBuffering {
                        specialized: false,
                        tile_matmul,
                    },
                    true,
                    None,
                ),
                (
                    FusedMatmulSelector::DoubleBuffering {
                        specialized: true,
                        tile_matmul,
                    },
                    true,
                    Some(&odd),
                ),
            ] {
                let priority_within_group =
                    |key: &FusedMatmulAutotuneKey, double_buf: bool| match double_buf {
                        false => PRIORITY_MAX,
                        true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                    };
                let mut tunable = Tunable::new(selector.name(), move |input| {
                    tune_fused::<R>(input, selector)
                })
                .group(&accelerated, move |key| {
                    priority_within_group(key, double_buf)
                });

                if let Some(group) = extra_group {
                    tunable =
                        tunable.group(group, move |key| priority_within_group(key, double_buf));
                }
                set = set.with(tunable);
            }
        }

        set
    });

    TUNER.execute(
        &CubeTuneId::new(&optimization.info.client, &optimization.info.device),
        &optimization.info.client.clone(),
        tunables,
        TuneInput::new(context, optimization),
    );
}

pub(crate) fn create_key<R: Runtime>(
    input: &TuneInput<R, MatmulOptimizationTuneArg<R>>,
) -> FusedMatmulAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => panic!("Not supported when generating key"),
    };

    let lhs = context.tensors.get(&opt.info.matmul.op.lhs.id).unwrap();
    let rhs = context.tensors.get(&opt.info.matmul.op.rhs.id).unwrap();
    let out = context.tensors.get(&opt.info.matmul.op.out.id).unwrap();

    let lhs_strides = context
        .handles
        .get_handle(&lhs.id, &burn_ir::TensorStatus::ReadOnly)
        .strides
        .clone();
    let rhs_strides = context
        .handles
        .get_handle(&rhs.id, &burn_ir::TensorStatus::ReadOnly)
        .strides
        .clone();

    let key = MatmulAutotuneKey::generate(
        &opt.info.client,
        &lhs.shape,
        &rhs.shape,
        &lhs_strides,
        &rhs_strides,
        lhs.dtype.into(),
        rhs.dtype.into(),
        out.dtype.into(),
        opt.info.matmul.lhs.scheme(),
        opt.info.matmul.rhs.scheme(),
    );
    FusedMatmulAutotuneKey::new(key, opt.info.num_output_buffers(), opt.info.num_ops_fused())
}

fn input_gen<R: Runtime>(
    _key: &FusedMatmulAutotuneKey,
    input: &TuneInput<R, MatmulOptimizationTuneArg<R>>,
) -> TuneInput<R, MatmulOptimizationTuneArg<R>> {
    input.clone()
}

fn tune_fused<R: Runtime>(
    input: TuneInput<R, MatmulOptimizationTuneArg<R>>,
    selector: FusedMatmulSelector,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => match optimization.execute_fused(context, selector) {
            Ok(out) => Ok(out),
            Err(_) => {
                return tune_fallback::<R>(input);
            }
        },
        TuneContext::Fork(mut fork) => optimization.execute_fused(&mut fork.as_context(), selector),
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_fallback<R: Runtime>(
    input: TuneInput<R, MatmulOptimizationTuneArg<R>>,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    Ok(match context {
        TuneContext::Original(context) => optimization.execute_fallback(context),
        TuneContext::Fork(mut fork) => optimization.execute_fallback(&mut fork.as_context()),
    })
}
