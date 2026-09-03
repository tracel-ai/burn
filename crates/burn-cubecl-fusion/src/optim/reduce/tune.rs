use super::optimization::ReduceOptimizationTuneArg;
use crate::{
    CubeFusionHandle,
    engine::trace::TuneOutput,
    tune::{FusionInputGen, TuneInput},
};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_fusion::stream::Context;
use cubecl::{
    AutotuneKey, CubeTuneId,
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
pub fn fused_reduce_autotune(
    arg: ReduceOptimizationTuneArg,
    context: &mut Context<CubeFusionHandle>,
) {
    static TUNER: LocalTuner<FusedReduceAutotuneKey, CubeTuneId> = local_tuner!();

    let tune_id = CubeTuneId::new(&arg.info.client, &arg.info.device);
    let tunables = TUNER.init(&tune_id, || {
        const PRIORITY_MAX: i8 = 2;
        const PRIORITY_MIN: i8 = 1;

        let mut set = TunableSet::new(create_key, FusionInputGen);
        let group = TuneGroup::<FusedReduceAutotuneKey>::new("fused_reduce", |_key| PRIORITY_MAX);

        // Fallback implementation for robustness.
        set = set.with(Tunable::new("fused_reduce_fallback", tune_fallback));

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
            let tunable = Tunable::new(name, move |input| tune_reduce(input, &strategy)).group(
                &group,
                move |key| match props {
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
                },
            );

            set = set.with(tunable);
        }

        set
    });

    TUNER.execute(
        &tune_id,
        &arg.info.client.clone(),
        tunables,
        TuneInput::new(context, arg),
    );
}

/// Creates the autotune key by extracting tensor metadata and fusion block statistics.
pub(crate) fn create_key(input: &TuneInput<ReduceOptimizationTuneArg>) -> FusedReduceAutotuneKey {
    let opt = input.optimization();
    assert!(
        input.is_original(),
        "Forked context not supported for key generation"
    );
    let tensors = input.tensors();

    let input_tensor = tensors.get(&opt.info.reduce.op.input.id).unwrap();
    let out_tensor = tensors.get(&opt.info.reduce.op.out.id).unwrap();
    let acc = opt.info.reduce.acc.into_elem();

    let key = ReduceAutotuneKey::generate(
        dtype_to_elem_type(input_tensor.dtype),
        dtype_to_elem_type(out_tensor.dtype),
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

/// Executes a fused reduction optimization.
fn tune_reduce(
    input: TuneInput<ReduceOptimizationTuneArg>,
    strategy: &RoutineStrategy,
) -> Result<TuneOutput, String> {
    input
        .execute(|ctx, opt| opt.execute_fused(ctx, strategy.clone()))
        .map_err(|e| format!("{e:?}"))
}

/// Executes the fallback path for a reduction optimization.
fn tune_fallback(input: TuneInput<ReduceOptimizationTuneArg>) -> Result<TuneOutput, String> {
    input.execute(|ctx, opt| {
        opt.execute_fallback(ctx);
    });
    Ok(TuneOutput::UnChecked)
}
