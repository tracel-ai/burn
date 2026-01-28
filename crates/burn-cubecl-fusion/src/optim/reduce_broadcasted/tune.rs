use crate::{
    CubeFusionHandle,
    engine::trace::TuneOutput,
    optim::reduce_broadcasted::ReduceBlockOptimArg,
    tune::{TuneContext, TuneInput},
};
use burn_fusion::stream::Context;
use cubecl::{
    AutotuneKey, CubeElement, CubeTuneId, Runtime,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};
use cubek::reduce::{
    launch::{RoutineStrategy, tune_key::ReduceAutotuneKey},
    routines::{BlueprintStrategy, unit::UnitStrategy},
};
use serde::{Deserialize, Serialize};

use super::optimization::ReduceBroadcastedOptimizationTuneArg;

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct FusedBroadcastedReduceAutotuneKey {
    reduce_key: ReduceAutotuneKey,
    #[autotune(anchor)]
    fuse_num_reads: usize,
    #[autotune(anchor)]
    fuse_num_writes: usize,
    #[autotune(anchor)]
    fuse_num_ops: usize,
    fuse_num_blocks: usize,
}

/// Executes autotune on reduce operations
pub fn fused_broadcasted_reduce_autotune<R: Runtime, BT: CubeElement>(
    arg: ReduceBroadcastedOptimizationTuneArg<R>,
    context: &mut Context<CubeFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedBroadcastedReduceAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        const PRIORITY_MAX: i8 = 2;

        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>);

        let group = TuneGroup::<FusedBroadcastedReduceAutotuneKey>::new(
            "fused_reduce_broadcasted",
            |_key| PRIORITY_MAX,
        );

        // First one should always work.
        set = set.with(Tunable::new(
            "fused_reduce_broadcasted_fallback",
            // tune_fallback::<R, BT>,
            move |input| {
                tune_reduce::<R, BT>(
                    input,
                    &RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                )
            },
        ));

        set = set.with(
            Tunable::new("fused_reduce_broadcasted_unit", move |input| {
                tune_reduce::<R, BT>(
                    input,
                    &RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                )
            })
            .group(&group, |_| PRIORITY_MAX),
        );

        set
    });

    TUNER.execute(
        &CubeTuneId::new(&arg.client, &arg.device),
        &arg.client.clone(),
        tunables,
        TuneInput::new(context, arg),
    );
}

pub(crate) fn create_key<R: Runtime>(
    input: &TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
) -> FusedBroadcastedReduceAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => panic!("Not supported when generating key"),
    };

    let info = match &opt.fallbacks[0] {
        ReduceBlockOptimArg::Reduce(reduce) => reduce.info.clone(),
        ReduceBlockOptimArg::Elemwise(_) => panic!(),
    };
    let input = context.tensors.get(&info.reduce.op.input.id).unwrap();
    let out = context.tensors.get(&info.reduce.op.out.id).unwrap();
    let acc = info.reduce.acc.into_elem();
    let key = ReduceAutotuneKey::generate(
        input.dtype.into(),
        out.dtype.into(),
        acc,
        &input.shape.dims,
        info.reduce.axis == input.shape.rank() - 1,
        info.reduce.axis,
    );
    let read = &info.trace.blocks[0];
    let write = &info.trace.blocks[1];

    FusedBroadcastedReduceAutotuneKey::new(
        key,
        read.reads.len() + write.reads.len(),
        read.writes.len() + write.writes.len(),
        read.ops.len() + write.ops.len(),
        opt.fallbacks.len(),
    )
}

fn input_gen<R: Runtime>(
    _key: &FusedBroadcastedReduceAutotuneKey,
    input: &TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
) -> TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>> {
    input.clone()
}

#[allow(unused)]
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

    match context {
        TuneContext::Original(context) => optimization.execute_fallback::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fallback::<BT>(&mut context_owned.as_context())
        }
    };
    Ok(TuneOutput::UnChecked(std::marker::PhantomData))
}
