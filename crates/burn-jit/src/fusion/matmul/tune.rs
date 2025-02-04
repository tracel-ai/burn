use crate::{
    fusion::{
        tune::{TuneContext, TuneInput},
        JitFusionHandle,
    },
    kernel::matmul::MatmulAutotuneKey,
    BoolElement, JitRuntime, JitTuneId,
};
use burn_fusion::stream::Context;
use cubecl::{
    tune::{local_tuner, LocalTuner, TunableSet},
    AutotuneKey,
};
use serde::{Deserialize, Serialize};

use super::optimization::MatmulOptimization;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct FusedMatmulAutotuneKey {
    matmul_key: MatmulAutotuneKey,
    #[autotune(anchor)]
    num_out_buffers: usize,
    #[autotune(anchor)]
    num_ops: usize,
}

/// Executes autotune on matmul operations
pub fn fused_matmul_autotune<R: JitRuntime, BT: BoolElement>(
    optimization: &MatmulOptimization<R>,
    context: &mut Context<JitFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedMatmulAutotuneKey, JitTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R>, input_gen::<R>)
        .with_tunable(tune_standard_fused::<R, BT>)
        .with_tunable(tune_specialized_fused::<R, BT>)
        .with_tunable(tune_pipelined_fused::<R, BT>)
        .with_tunable(tune_fallback::<R, BT>);

    TUNER.execute(
        &JitTuneId::new::<R>(&optimization.device),
        &optimization.client,
        &tunables,
        TuneInput::new(context, optimization),
    );
}

pub(crate) fn create_key<R: JitRuntime>(
    input: &TuneInput<R, MatmulOptimization<R>>,
) -> FusedMatmulAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => panic!("Not supported when generating key"),
    };

    let lhs = context.tensors.get(&opt.matmul_standard.op.lhs.id).unwrap();
    let rhs = context.tensors.get(&opt.matmul_standard.op.rhs.id).unwrap();
    let out = context.tensors.get(&opt.matmul_standard.op.out.id).unwrap();

    let key = MatmulAutotuneKey::from_shape(
        &lhs.shape.clone().into(),
        &rhs.shape.clone().into(),
        out.dtype,
    );
    FusedMatmulAutotuneKey::new(key, opt.num_output_buffers(), opt.num_ops_fused())
}

fn input_gen<R: JitRuntime>(
    _key: &FusedMatmulAutotuneKey,
    input: &TuneInput<R, MatmulOptimization<R>>,
) -> TuneInput<R, MatmulOptimization<R>> {
    input.clone()
}

fn tune_standard_fused<R: JitRuntime, BT: BoolElement>(
    input: TuneInput<R, MatmulOptimization<R>>,
) -> Result<(), String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => optimization.execute_standard_fused::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_standard_fused::<BT>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_specialized_fused<R: JitRuntime, BT: BoolElement>(
    input: TuneInput<R, MatmulOptimization<R>>,
) -> Result<(), String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => optimization.execute_specialized_fused::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_specialized_fused::<BT>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_pipelined_fused<R: JitRuntime, BT: BoolElement>(
    input: TuneInput<R, MatmulOptimization<R>>,
) -> Result<(), String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => optimization.execute_pipelined_fused::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_pipelined_fused::<BT>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_fallback<R: JitRuntime, BT: BoolElement>(
    input: TuneInput<R, MatmulOptimization<R>>,
) -> Result<(), String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => optimization.execute_fallback::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fallback::<BT>(&mut context_owned.as_context())
        }
    };

    Ok(())
}
