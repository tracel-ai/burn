use super::optimization::{MatmulOptimizationTuneArg, MatmulVariantSelection};
use crate::{
    CubeFusionHandle,
    engine::trace::TuneOutput,
    optim::matmul::optimization::{
        DoubleBuffering, DoubleBufferingMma, DoubleUnit, DoubleVecMat, Ordered, OrderedMma, Simple,
        SimpleMma, SimpleMultiRows, SimpleMultiRowsMma, SimpleUnit, SimpleVecMat, Specialized,
        SpecializedMma,
    },
    tune::{TuneContext, TuneInput},
};
use burn_fusion::stream::Context;
use burn_tensor::DType;
use cubecl::{
    AutotuneKey, CubeElement, CubeTuneId, Runtime,
    matmul::{
        components::MatmulKind,
        tune_key::{
            MatmulAutotuneKey, MatmulElemType, MatmulGlobalScale, should_tune_double_buffering,
        },
    },
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
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

        let cmma = TuneGroup::<FusedMatmulAutotuneKey>::new(|key| {
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

        let mma = TuneGroup::<FusedMatmulAutotuneKey>::new(|key| {
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

        let odd = TuneGroup::<FusedMatmulAutotuneKey>::new(|key| {
            if key.matmul_key.definition.lhs_pow2_factor == 0
                || key.matmul_key.definition.rhs_pow2_factor == 0
            {
                PRIORITY_MAX
            } else {
                PRIORITY_MIN
            }
        });

        let unit = TuneGroup::<FusedMatmulAutotuneKey>::new(|key| {
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

        TunableSet::new(create_key::<R>, input_gen::<R>)
            .with(Tunable::new(tune_fallback::<R, BT>)) // First one should always work.
            .with(Tunable::new(tune_fused::<R, BT, SimpleUnit>).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(tune_fused::<R, BT, SimpleVecMat>).group(&unit, |_| PRIORITY_MAX))
            .with(Tunable::new(tune_fused::<R, BT, DoubleVecMat>).group(&unit, |_| PRIORITY_MAX))
            .with(
                Tunable::new(tune_fused::<R, BT, DoubleUnit>).group(&unit, |key| {
                    double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH)
                }),
            )
            .with(Tunable::new(tune_fused::<R, BT, Simple>).group(&cmma, |_| PRIORITY_MAX))
            .with(Tunable::new(tune_fused::<R, BT, SimpleMma>).group(&mma, |_| PRIORITY_MAX))
            .with(Tunable::new(tune_fused::<R, BT, SimpleMultiRows>).group(&cmma, |_| PRIORITY_MAX))
            .with(
                Tunable::new(tune_fused::<R, BT, SimpleMultiRowsMma>).group(&mma, |_| PRIORITY_MAX),
            )
            // Ordered should be tried most of the time.
            .with(Tunable::new(tune_fused::<R, BT, Ordered>).group(&cmma, |_| PRIORITY_MAX))
            .with(Tunable::new(tune_fused::<R, BT, OrderedMma>).group(&mma, |_| PRIORITY_MAX))
            .with(
                Tunable::new(tune_fused::<R, BT, Specialized>)
                    .group(&cmma, |key| {
                        double_buffering_priority(key, PRIORITY_HIGH, PRIORITY_MIN)
                    })
                    .group(&odd, |_| PRIORITY_MAX),
            )
            .with(
                Tunable::new(tune_fused::<R, BT, SpecializedMma>)
                    .group(&cmma, |key| {
                        double_buffering_priority(key, PRIORITY_HIGH, PRIORITY_MIN)
                    })
                    .group(&odd, |_| PRIORITY_MAX),
            )
            .with(
                Tunable::new(tune_fused::<R, BT, DoubleBuffering>)
                    .group(&cmma, |key| {
                        double_buffering_priority(key, PRIORITY_HIGH, PRIORITY_MEDIUM)
                    })
                    .group(&odd, |_| PRIORITY_MAX),
            )
            .with(
                Tunable::new(tune_fused::<R, BT, DoubleBufferingMma>)
                    .group(&cmma, |key| {
                        double_buffering_priority(key, PRIORITY_HIGH, PRIORITY_MEDIUM)
                    })
                    .group(&odd, |_| PRIORITY_MAX),
            )
    });

    TUNER.execute(
        &CubeTuneId::new::<R>(&optimization.info.client, &optimization.info.device),
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

    let lhs = context
        .tensors
        .get(&opt.info.variants.simple.op.lhs.id)
        .unwrap();
    let rhs = context
        .tensors
        .get(&opt.info.variants.simple.op.rhs.id)
        .unwrap();
    let out = context
        .tensors
        .get(&opt.info.variants.simple.op.out.id)
        .unwrap();

    let lhs_strides = context
        .handles
        .get_handle(&lhs.id, &burn_ir::TensorStatus::ReadOnly)
        .strides;
    let rhs_strides = context
        .handles
        .get_handle(&rhs.id, &burn_ir::TensorStatus::ReadOnly)
        .strides;

    let key = MatmulAutotuneKey::generate::<R>(
        &opt.info.client,
        &lhs.shape.dims,
        &rhs.shape.dims,
        &lhs_strides,
        &rhs_strides,
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
    );
    FusedMatmulAutotuneKey::new(key, opt.info.num_output_buffers(), opt.info.num_ops_fused())
}

fn input_gen<R: Runtime>(
    _key: &FusedMatmulAutotuneKey,
    input: &TuneInput<R, MatmulOptimizationTuneArg<R>>,
) -> TuneInput<R, MatmulOptimizationTuneArg<R>> {
    input.clone()
}

fn tune_fused<R: Runtime, BT: CubeElement, S: MatmulVariantSelection>(
    input: TuneInput<R, MatmulOptimizationTuneArg<R>>,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => match optimization.execute_fused::<BT, S>(context) {
            Ok(out) => Ok(out),
            Err(_) => {
                return tune_fallback::<R, BT>(input);
            }
        },
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused::<BT, S>(&mut context_owned.as_context())
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
