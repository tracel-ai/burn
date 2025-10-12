use crate::{
    CubeFusionHandle,
    matmul::optimization::{
        DoubleBuffering, DoubleUnit, DoubleVecMat, Ordered, Simple, SimpleMultiRows, SimpleUnit,
        SimpleVecMat, Specialized,
    },
    shared::trace::TuneOutput,
    tune::{TuneContext, TuneInput},
};
use burn_fusion::stream::Context;
use cubecl::{
    AutotuneKey, CubeElement, CubeTuneId, Runtime,
    matmul::{
        components::MatmulKind,
        tune_key::{MatmulAutotuneKey, MatmulGlobalScale, should_tune_double_buffering},
    },
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};
use serde::{Deserialize, Serialize};

use super::optimization::{MatmulOptimizationTuneArg, MatmulVariantSelection};

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
        const PRIORITY_MAX: u8 = 3;
        const PRIORITY_HIGH: u8 = 2;
        const PRIORITY_MEDIUM: u8 = 1;
        const PRIORITY_MIN: u8 = 0;

        let cmma = TuneGroup::<FusedMatmulAutotuneKey>::new(|key| {
            if matches!(key.matmul_key.analysis.kind, MatmulKind::General) {
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

        fn double_buffering_priority(key: &FusedMatmulAutotuneKey, max: u8, min: u8) -> u8 {
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
            .with(Tunable::new(tune_fused::<R, BT, SimpleMultiRows>).group(&cmma, |_| PRIORITY_MAX))
            // Ordered should be tried most of the time.
            .with(Tunable::new(tune_fused::<R, BT, Ordered>).group(&cmma, |_| PRIORITY_MAX))
            .with(
                Tunable::new(tune_fused::<R, BT, Specialized>)
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
        lhs.dtype.into(),
        rhs.dtype.into(),
        out.dtype.into(),
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
