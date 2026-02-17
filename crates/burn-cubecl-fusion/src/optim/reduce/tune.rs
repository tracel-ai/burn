use super::optimization::ReduceOptimizationTuneArg;
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

/// Autotune key for standard fused reduction operations.
///
/// Records metadata about the fusion graph (IO and ops) alongside
/// the core reduction parameters to ensure stable kernel selection.
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

/// Executes autotuning for fused reduction operations.
///
/// This tuner evaluates different hardware-specific strategies (Plane, Cube, Unit)
/// and assigns priorities based on the `vector_count` of the reduction.
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

        // Fallback implementation for robustness.
        set = set.with(Tunable::new(
            "fused_reduce_fallback",
            tune_fallback::<R, BT>,
        ));

        // Define properties to categorize hardware strategies.
        enum ReduceProps {
            GreatWithLowReduceCount,
            GreatWithHighReduceCount,
            Balanced,
        }

        let strategies = [
            (
                "fused_unit",
                RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                ReduceProps::GreatWithHighReduceCount,
            ),
            (
                "fused_plane",
                RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                    independent: true,
                })),
                ReduceProps::Balanced,
            ),
            (
                "fused_cube",
                RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                    // Two steps reduction doesn't work with fuse-on-write, we can't activate plane
                    // when using the cube algo.
                    use_planes: false,
                })),
                ReduceProps::GreatWithLowReduceCount,
            ),
        ];

        for (name, strategy, props) in strategies {
            let tunable = Tunable::new(name, move |input| tune_reduce::<R, BT>(input, &strategy))
                .group(&group, move |key| match props {
                    ReduceProps::GreatWithLowReduceCount => {
                        if key.reduce_key.vector_count < 128 {
                            PRIORITY_MAX
                        } else {
                            PRIORITY_MIN
                        }
                    }
                    ReduceProps::GreatWithHighReduceCount => {
                        if key.reduce_key.vector_count > 64 {
                            PRIORITY_MAX
                        } else {
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

/// Creates the autotune key by extracting tensor metadata and fusion block statistics.
pub(crate) fn create_key<R: Runtime>(
    input: &TuneInput<R, ReduceOptimizationTuneArg<R>>,
) -> FusedReduceAutotuneKey {
    let opt = input.optimization();
    let context = match input.context() {
        TuneContext::Original(context) => context,
        TuneContext::Fork(_) => panic!("Forked context not supported for key generation"),
    };

    let input_tensor = context.tensors.get(&opt.info.reduce.op.input.id).unwrap();
    let out_tensor = context.tensors.get(&opt.info.reduce.op.out.id).unwrap();
    let acc = opt.info.reduce.acc.into_elem();

    let key = ReduceAutotuneKey::generate(
        input_tensor.dtype.into(),
        out_tensor.dtype.into(),
        acc,
        &input_tensor.shape,
        opt.info.reduce.axis == input_tensor.shape.rank() - 1,
        opt.info.reduce.axis,
    );

    // Assume the fusion contains at least a read and a write block.
    let read_block = &opt.info.trace.blocks[0];
    let write_block = &opt.info.trace.blocks[1];

    FusedReduceAutotuneKey::new(
        key,
        read_block.reads.len() + write_block.reads.len(),
        read_block.writes.len() + write_block.writes.len(),
        read_block.ops.len() + write_block.ops.len(),
    )
}

/// Identity generator for tuning inputs.
fn input_gen<R: Runtime>(
    _key: &FusedReduceAutotuneKey,
    input: &TuneInput<R, ReduceOptimizationTuneArg<R>>,
) -> TuneInput<R, ReduceOptimizationTuneArg<R>> {
    input.clone()
}

/// Executes a fused reduction optimization.
fn tune_reduce<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceOptimizationTuneArg<R>>,
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

/// Executes the fallback path for a reduction optimization.
fn tune_fallback<R: Runtime, BT: CubeElement>(
    input: TuneInput<R, ReduceOptimizationTuneArg<R>>,
) -> Result<TuneOutput<R>, String> {
    let optimization = input.optimization();

    match input.context() {
        TuneContext::Original(context) => optimization.execute_fallback::<BT>(context),
        TuneContext::Fork(mut context_owned) => {
            optimization.execute_fallback::<BT>(&mut context_owned.as_context())
        }
    };

    Ok(TuneOutput::UnChecked(std::marker::PhantomData))
}
