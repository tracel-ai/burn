use crate::{
    CubeFusionHandle,
    reduce::optimization::{Normal, Plane, ReduceVariantSelection, SharedPlane},
    shared::trace::TuneOutput,
    tune::{TuneContext, TuneInput},
};
use burn_fusion::stream::Context;
use cubecl::{
    AutotuneKey, CubeElement, CubeTuneId, Runtime,
    reduce::tune_key::ReduceAutotuneKey,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use serde::{Deserialize, Serialize};

use super::optimization::ReduceOptimizationTuneArg;

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
    arg: ReduceOptimizationTuneArg<R>,
    context: &mut Context<CubeFusionHandle<R>>,
) {
    static TUNER: LocalTuner<FusedReduceAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R>, input_gen::<R>)
            .with(Tunable::new(tune_fallback::<R, BT>)) // First one should always work.
            .with(Tunable::new(tune_reduce::<R, BT, Normal>))
            .with(Tunable::new(tune_reduce::<R, BT, Plane>))
            .with(Tunable::new(tune_reduce::<R, BT, SharedPlane>))
            .with(Tunable::new(tune_reduce_on_read::<R, BT, Normal>))
            .with(Tunable::new(tune_reduce_on_read::<R, BT, Plane>))
            .with(Tunable::new(tune_reduce_on_read::<R, BT, SharedPlane>))
    });

    TUNER.execute(
        &CubeTuneId::new::<R>(&arg.info.client, &arg.info.device),
        &arg.info.client.clone(),
        tunables,
        TuneInput::new(context, arg),
    );
}

pub(crate) fn create_key<R: Runtime>(
    input: &TuneInput<R, ReduceOptimizationTuneArg<R>>,
) -> FusedReduceAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => panic!("Not supported when generating key"),
    };

    let input = context
        .tensors
        .get(&opt.info.variants.reduce.op.input.id)
        .unwrap();
    let out = context
        .tensors
        .get(&opt.info.variants.reduce.op.out.id)
        .unwrap();
    let acc = opt.info.variants.reduce.acc.into_elem();
    let key = ReduceAutotuneKey::generate(
        input.dtype.into(),
        out.dtype.into(),
        acc,
        &input.shape.dims,
        opt.info.variants.reduce.axis == input.shape.rank() - 1,
        opt.info.variants.reduce.axis,
    );
    let read = &opt.info.trace.blocks[0];
    let write = &opt.info.trace.blocks[1];

    FusedReduceAutotuneKey::new(
        key,
        read.reads.len() + write.reads.len(),
        read.writes.len() + write.writes.len(),
        read.ops.len() + write.ops.len(),
    )
}

fn input_gen<R: Runtime>(
    _key: &FusedReduceAutotuneKey,
    input: &TuneInput<R, ReduceOptimizationTuneArg<R>>,
) -> TuneInput<R, ReduceOptimizationTuneArg<R>> {
    input.clone()
}

fn tune_reduce<R: Runtime, BT: CubeElement, S: ReduceVariantSelection>(
    input: TuneInput<R, ReduceOptimizationTuneArg<R>>,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => optimization.execute_fused::<BT, S>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused::<BT, S>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_reduce_on_read<R: Runtime, BT: CubeElement, S: ReduceVariantSelection>(
    input: TuneInput<R, ReduceOptimizationTuneArg<R>>,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();
    let context = input.context();

    match context {
        TuneContext::Original(context) => optimization.execute_fused_on_read::<BT, S>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fused_on_read::<BT, S>(&mut context_owned.as_context())
        }
    }
    .map_err(|e| format!("{e:?}"))
}

fn tune_fallback<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceOptimizationTuneArg<R>>,
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
