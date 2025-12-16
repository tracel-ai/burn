use super::optimization::MatmulOptimizationTuneArg;
use crate::{
    CubeFusionHandle,
    engine::trace::TuneOutput,
    optim::matmul::FusedMatmulSelector,
    tune::{TuneContext, TuneInput},
};
use burn_fusion::stream::Context;
use burn_std::DType;
use cubecl::{
    AutotuneKey, CubeElement, CubeTuneId, Runtime,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};
use cubek::matmul::{
    definition::{MatmulElemType, MatmulKind},
    launch::{
        AcceleratedTileKind, MatmulAutotuneKey, MatmulGlobalScale, should_tune_double_buffering,
    },
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
pub fn fused_matmul_autotune<R: Runtime, BT: CubeElement>(
    optimization: MatmulOptimizationTuneArg<R>,
    context: &mut Context<CubeFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedMatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: i8 = 3;
        const PRIORITY_HIGH: i8 = 2;
        const PRIORITY_MEDIUM: i8 = 1;
        const PRIORITY_MIN: i8 = 0;

        let cmma = TuneGroup::<FusedMatmulAutotuneKey>::new("cmma", |key| {
            if matches!(
                key.matmul_key.analysis.kind,
                MatmulKind::General
                // Those variants are just because the unit alternatives aren't very good yet.
                | MatmulKind::VecMat | MatmulKind::MatVec
            ) {
                PRIORITY_MAX
            } else {
                PRIORITY_MEDIUM
            }
        });

        let mma = TuneGroup::<FusedMatmulAutotuneKey>::new("mma", |key| {
            if matches!(
                key.matmul_key.analysis.kind,
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

        let odd = TuneGroup::<FusedMatmulAutotuneKey>::new("odd", |key| {
            if key.matmul_key.definition.lhs_pow2_factor == 0
                || key.matmul_key.definition.rhs_pow2_factor == 0
            {
                PRIORITY_MAX
            } else {
                PRIORITY_MIN
            }
        });

        let unit = TuneGroup::<FusedMatmulAutotuneKey>::new("unit", |key| {
            if !matches!(key.matmul_key.analysis.kind, MatmulKind::General)
                || matches!(
                    key.matmul_key.analysis.scale_global,
                    MatmulGlobalScale::Small
                )
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

        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>).with(Tunable::new(
            "fused_matmul_fallback",
            tune_fallback::<R, BT>,
        )); // First one should always work.

        // Unit matmuls
        for (selector, double_buf) in [
            (FusedMatmulSelector::SimpleUnit, false),
            (FusedMatmulSelector::DoubleUnit, true),
            (FusedMatmulSelector::SimpleVecMat, false),
            (FusedMatmulSelector::DoubleVecMat, true),
        ] {
            set = set.with(
                Tunable::new(selector.name(), move |input| {
                    tune_fused::<R, BT>(input, selector)
                })
                .group(&unit, move |key| match double_buf {
                    true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                    false => PRIORITY_MAX,
                }),
            );
        }

        // Accelerated matmuls
        for (tile_matmul, group) in [
            (AcceleratedTileKind::Cmma, &cmma),
            (AcceleratedTileKind::Mma, &mma),
        ] {
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
                let mut tunable = Tunable::new(selector.name(), move |input| {
                    tune_fused::<R, BT>(input, selector)
                })
                .group(group, move |key| match double_buf {
                    true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                    false => PRIORITY_MAX,
                });
                if let Some(group) = extra_group {
                    tunable = tunable.group(group, |_| PRIORITY_MAX);
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
        .strides;
    let rhs_strides = context
        .handles
        .get_handle(&rhs.id, &burn_ir::TensorStatus::ReadOnly)
        .strides;

    let key = MatmulAutotuneKey::generate(
        &opt.info.client,
        &lhs.shape.dims,
        &rhs.shape.dims,
        &lhs_strides,
        &rhs_strides,
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
    );
    FusedMatmulAutotuneKey::new(key, opt.info.num_output_buffers(), opt.info.num_ops_fused())
}

fn input_gen<R: Runtime>(
    _key: &FusedMatmulAutotuneKey,
    input: &TuneInput<R, MatmulOptimizationTuneArg<R>>,
) -> TuneInput<R, MatmulOptimizationTuneArg<R>> {
    input.clone()
}

fn tune_fused<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, MatmulOptimizationTuneArg<R>>,
    selector: FusedMatmulSelector,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => match optimization.execute_fused::<BT>(context, selector)
        {
            Ok(out) => Ok(out),
            Err(_) => {
                return tune_fallback::<R, BT>(input);
            }
        },
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused::<BT>(&mut context_owned.as_context(), selector)
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_fallback<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, MatmulOptimizationTuneArg<R>>,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    Ok(match context {
        TuneContext::Original(context) => optimization.execute_fallback::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fallback::<BT>(&mut context_owned.as_context())
        }
    })
}
