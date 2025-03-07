use crate::{
    tune::{TuneContext, TuneInput},
    CubeFusionHandle,
};
use burn_fusion::stream::Context;
use cubecl::{
    reduce::tune_key::ReduceAutotuneKey,
    tune::{local_tuner, LocalTuner, TunableSet},
    AutotuneKey, CubeElement, CubeTuneId, Runtime,
};
use serde::{Deserialize, Serialize};

use super::optimization::ReduceOptimization;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct FusedReduceAutotuneKey {
    reduce_key: ReduceAutotuneKey,
    #[autotune(anchor)]
    num_out_buffers: usize,
    #[autotune(anchor)]
    num_ops: usize,
    #[autotune(anchor)]
    vect_count: usize,
}

/// Executes autotune on reduce operations
pub fn fused_reduce_autotune<R: Runtime, BT: CubeElement>(
    optimization: &ReduceOptimization<R>,
    context: &mut Context<CubeFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedReduceAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R>, input_gen::<R>)
        .with_tunable(tune_reduce::<R, BT>)
        .with_tunable(tune_reduce_plane::<R, BT>)
        .with_tunable(tune_reduce_shared_plane::<R, BT>)
        .with_tunable(tune_fallback::<R, BT>);

    TUNER.execute(
        &CubeTuneId::new::<R>(&optimization.device),
        &optimization.client,
        &tunables,
        TuneInput::new(context, optimization),
    );
}

pub(crate) fn create_key<R: Runtime>(
    input: &TuneInput<R, ReduceOptimization<R>>,
) -> FusedReduceAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => panic!("Not supported when generating key"),
    };

    let input = context.tensors.get(&opt.reduce.op.input.id).unwrap();
    let out = context.tensors.get(&opt.reduce.op.out.id).unwrap();
    let key = ReduceAutotuneKey::generate_without_strides(
        out.dtype.into(),
        &input.shape,
        opt.reduce.axis,
    );
    let vect = opt.trace_read.vect(context, &opt.reduce);
    let vect = vect.iter().map(|v| v.1.line_size() as usize).sum::<usize>();

    FusedReduceAutotuneKey::new(key, opt.num_output_buffers(), opt.num_ops_fused(), vect)
}

fn input_gen<R: Runtime>(
    _key: &FusedReduceAutotuneKey,
    input: &TuneInput<R, ReduceOptimization<R>>,
) -> TuneInput<R, ReduceOptimization<R>> {
    input.clone()
}

fn tune_reduce<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceOptimization<R>>,
) -> Result<(), String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => optimization.execute_fused_reduce::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused_reduce::<BT>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_reduce_plane<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceOptimization<R>>,
) -> Result<(), String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => optimization.execute_fused_reduce_plane::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused_reduce_plane::<BT>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_reduce_shared_plane<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceOptimization<R>>,
) -> Result<(), String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => {
            optimization.execute_fused_reduce_shared_plane::<BT>(context)
        }
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused_reduce_shared_plane::<BT>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_fallback<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceOptimization<R>>,
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
