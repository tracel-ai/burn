use super::optimization::ReduceBroadcastedOptimizationTuneArg;
use crate::{
    CubeFusionHandle,
    engine::trace::TuneOutput,
    optim::{reduce::ReduceOptimizationInfo, reduce_broadcasted::ReduceBlockOptimArg},
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

/// Autotune key for fused broadcasted reduction operations.
///
/// Captures the characteristics of the fusion (reads, writes, ops) to ensure
/// the best kernel is selected for specific fused graph shapes.
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

/// Executes the autotuning process for fused reduction operations.
///
/// This function initializes a local tuner and attempts multiple strategies
/// (fallback vs. unit strategy) to find the most efficient execution path.
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

        // Standard fallback implementation - guaranteed to work.
        set = set.with(Tunable::new(
            "fused_reduce_broadcasted_fallback",
            tune_fallback::<R, BT>,
        ));

        // Specialized unit strategy for fused reductions.
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

/// Generates the autotune key based on the current optimization context and trace blocks.
pub(crate) fn create_key<R: Runtime>(
    input: &TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
) -> FusedBroadcastedReduceAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => unreachable!("Forked context not supported for key generation"),
    };

    // The fusion must start with a reduction block to be valid here.
    let info = match &opt.fallbacks[0] {
        ReduceBlockOptimArg::Reduce(reduce) => &reduce.info,
        ReduceBlockOptimArg::Elemwise(_) => {
            unreachable!("Fusion must start with a reduction block")
        }
    };

    let key = generate_reduce_autotune_key(info, context);

    // Sum up complexity metrics across all blocks in the fused trace.
    let (mut num_reads, mut num_writes, mut num_ops) = (0, 0, 0);

    for block in opt.broadcasted.trace.blocks.iter() {
        num_reads += block.reads.len();
        num_writes += block.writes.len();
        num_ops += block.ops.len();
    }

    FusedBroadcastedReduceAutotuneKey::new(
        key,
        num_reads,
        num_writes,
        num_ops,
        info.trace.blocks.len(),
    )
}

/// Helper to generate the base reduction key (shapes, types, axes).
fn generate_reduce_autotune_key<R: Runtime>(
    info: &ReduceOptimizationInfo<R>,
    context: &Context<CubeFusionHandle<R>>,
) -> ReduceAutotuneKey {
    let input = context.tensors.get(&info.reduce.op.input.id).unwrap();
    let out = context.tensors.get(&info.reduce.op.out.id).unwrap();
    let acc = info.reduce.acc.into_elem();

    ReduceAutotuneKey::generate(
        input.dtype.into(),
        out.dtype.into(),
        acc,
        &input.shape.dims,
        info.reduce.axis == input.shape.rank() - 1, // Is it the last dimension?
        info.reduce.axis,
    )
}

/// Simple input generator that clones the input for the tuner.
fn input_gen<R: Runtime>(
    _key: &FusedBroadcastedReduceAutotuneKey,
    input: &TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
) -> TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>> {
    input.clone()
}

/// Executes a fused reduction using a specific routine strategy.
fn tune_reduce<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
    strategy: &RoutineStrategy,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();

    match input.context() {
        TuneContext::Original(context) => {
            optimization.execute_fused::<BT>(context, strategy.clone())
        }
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused::<BT>(&mut context_owned.as_context(), strategy.clone())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

/// Executes the fallback implementation for the reduction.
fn tune_fallback<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceBroadcastedOptimizationTuneArg<R>>,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();

    match input.context() {
        TuneContext::Original(context) => optimization.execute_fallback::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fallback::<BT>(&mut context_owned.as_context())
        }
    };

    // Fallback is often used as a baseline, returning unchecked output.
    Ok(TuneOutput::UnChecked(std::marker::PhantomData))
}
