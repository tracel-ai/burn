use crate::{
    CubeFusionHandle,
    engine::trace::TuneOutput,
    tune::{TuneContext, TuneInput},
};
use burn_fusion::stream::Context;
use cubecl::{
    AutotuneKey, CubeElement, CubeTuneId, Runtime,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};
use cubek::reduce::{
    launch::{RoutineStrategy, tune_key::ReduceAutotuneKey},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};
use serde::{Deserialize, Serialize};

use super::optimization::ReduceBroadcastedOptimizationTuneArg;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct FusedReduceAutotuneKey {
    reduce_key: ReduceAutotuneKey,
    #[autotune(anchor)]
    fuse_num_reads: usize,
    #[autotune(anchor)]
    fuse_num_writes: usize,
    #[autotune(anchor)]
    fuse_num_ops: usize,
}

/// Executes autotune on reduce operations
pub fn fused_reduce_autotune<R: Runtime, BT: CubeElement>(
    arg: ReduceBroadcastedOptimizationTuneArg<R>,
    context: &mut Context<CubeFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedReduceAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: i8 = 2;
        const PRIORITY_MIN: i8 = 1;

        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>);

        let group = TuneGroup::<FusedReduceAutotuneKey>::new("fused_reduce_broadcasted", |_key| {
            PRIORITY_MAX
        });

        // First one should always work.
        set = set.with(Tunable::new(
            "fused_reduce_broadcasted_fallback",
            tune_fallback::<R, BT>,
        ));

        set
    });

    TUNER.execute(
        &CubeTuneId::new(&arg.info.client, &arg.info.device),
        &arg.info.client.clone(),
        tunables,
        TuneInput::new(context, arg),
    );
}

pub(crate) fn create_key<R: Runtime>(
    input: &TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
) -> FusedReduceAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => panic!("Not supported when generating key"),
    };

    todo!()
}

fn input_gen<R: Runtime>(
    _key: &FusedReduceAutotuneKey,
    input: &TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
) -> TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>> {
    input.clone()
}

fn tune_reduce<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
    strategy: &RoutineStrategy,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => {
            optimization.execute_fused::<BT>(context, strategy.clone())
        }
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused::<BT>(&mut context_owned.as_context(), strategy.clone())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_fallback<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
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
