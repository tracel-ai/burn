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
        const PRIORITY_MAX: i8 = 2;
        const PRIORITY_MIN: i8 = 1;

        let mut set = TunableSet::new(create_key::<R>, input_gen::<R>);

        let group = TuneGroup::<FusedReduceAutotuneKey>::new("fused_reduce", |_key| PRIORITY_MAX);

        // First one should always work.
        set = set.with(Tunable::new(
            "fused_reduce_fallback",
            tune_fallback::<R, BT>,
        ));

        enum ReduceProps {
            GreatWithLowReduceCount,
            GreatWithHighReduceCount,
            Balanced,
        }

        for (name, strategy, props) in [
            (
                "fused_unit",
                RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                ReduceProps::GreatWithHighReduceCount,
            ),
            (
                "fused_plane",
                RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                    use_planes: true,
                })),
                ReduceProps::Balanced,
            ),
            (
                "fused_cube",
                RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                    independent: true,
                })),
                ReduceProps::GreatWithLowReduceCount,
            ),
        ] {
            let tunable = Tunable::new(name, move |input| tune_reduce::<R, BT>(input, &strategy))
                .group(&group, move |key| match props {
                    ReduceProps::GreatWithLowReduceCount => {
                        if key.reduce_key.vector_count < 128 {
                            PRIORITY_MAX
                        } else {
                            // When you have a high level of vector to reduce, it is normally
                            // better to use another routine.
                            PRIORITY_MIN
                        }
                    }
                    ReduceProps::GreatWithHighReduceCount => {
                        if key.reduce_key.vector_count > 64 {
                            PRIORITY_MAX
                        } else {
                            // Bellow 64 it is normally better to use another routine
                            PRIORITY_MIN
                        }
                    }
                    ReduceProps::Balanced => PRIORITY_MAX,
                });

            set = set.with(tunable);
        }

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
    input: &TuneInput<R, ReduceOptimizationTuneArg<R>>,
) -> FusedReduceAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => panic!("Not supported when generating key"),
    };

    let input = context.tensors.get(&opt.info.reduce.op.input.id).unwrap();
    let out = context.tensors.get(&opt.info.reduce.op.out.id).unwrap();
    let acc = opt.info.reduce.acc.into_elem();
    let key = ReduceAutotuneKey::generate(
        input.dtype.into(),
        out.dtype.into(),
        acc,
        &input.shape.dims,
        opt.info.reduce.axis == input.shape.rank() - 1,
        opt.info.reduce.axis,
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

fn tune_reduce<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceOptimizationTuneArg<R>>,
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
