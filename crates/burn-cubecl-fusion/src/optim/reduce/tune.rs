use super::optimization::{ReduceOptimizationInfo, ReduceOptimizationTuneArg, reduce_operation};
use crate::elector::elected_reduce;
use crate::{
    CubeFusionHandle,
    engine::trace::TuneOutput,
    tune::{FusionInputGen, TuneInput},
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use cubecl::{
    AutotuneKey, CubeTuneId,
    tile::Geometry,
    tune::{LocalTuner, Tunable, TunableSet, local_tuner},
};
use cubek::reduce::kernels::{ReduceElements, ReduceKey, ReduceRoutineProblem};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

/// Autotune key for standard fused reduction operations: the reduce's own key, as the backend's reduce
/// autotune states it, beside what the trace fuses around it.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct FusedReduceAutotuneKey {
    reduce_key: ReduceKey,
    #[autotune(anchor)]
    fuse_num_reads: usize,
    #[autotune(anchor)]
    fuse_num_writes: usize,
    #[autotune(anchor)]
    fuse_num_ops: usize,
}

/// Run the fused reduce: the routine the backend's reduce autotune elected for the unfused reduce,
/// reading through the trace's read block, then the write block, or the fallback, whichever the
/// autotune measured faster for its key.
pub fn fused_reduce_autotune(
    arg: ReduceOptimizationTuneArg,
    context: &mut Context<CubeFusionHandle>,
) {
    static TUNER: LocalTuner<FusedReduceAutotuneKey, CubeTuneId> = local_tuner!();

    // Until the backend's reduce autotune has elected a routine for the unfused reduce, the fused launch
    // has nothing to run. The fallback runs alone, outside a round, and its unfused reduce settles
    // that election; a round raced before it would settle the fused key on the fallback for good.
    let reduce_key = reduce_key(&arg.info, &context.tensors);
    if elected_reduce(&arg.info.client, &arg.info.device, &reduce_key).is_none() {
        arg.execute_fallback(context);
        return;
    }

    let tune_id = CubeTuneId::new(&arg.info.client, &arg.info.device);
    let tunables = TUNER.init(&tune_id, || {
        // The fallback first: it always runs, and its unfused reduce settles the key the fused
        // one elects its routine from.
        TunableSet::new(create_key, FusionInputGen)
            .with(Tunable::new("fused_reduce_fallback", tune_fallback))
            .with(Tunable::new("fused_reduce_tile", tune_fused))
    });

    TUNER.execute(
        &tune_id,
        &arg.info.client.clone(),
        tunables,
        TuneInput::new(context, arg),
    );
}

/// Creates the autotune key from the reduce's operands and the fusion blocks' statistics.
pub(crate) fn create_key(input: &TuneInput<ReduceOptimizationTuneArg>) -> FusedReduceAutotuneKey {
    let opt = input.optimization();
    assert!(
        input.is_original(),
        "Forked context not supported for key generation"
    );
    // Assume the fusion contains at least a read and a write block.
    let read_block = &opt.info.trace.blocks[0];
    let write_block = &opt.info.trace.blocks[1];

    FusedReduceAutotuneKey::new(
        reduce_key(&opt.info, input.tensors()),
        read_block.reads.len() + write_block.reads.len(),
        read_block.writes.len() + write_block.writes.len(),
        read_block.ops.len() + write_block.ops.len(),
    )
}

/// The unfused reduce's key, as the backend's reduce autotune states it: the problem the operands state,
/// not the widths they are read at.
fn reduce_key(info: &ReduceOptimizationInfo, tensors: &HashMap<TensorId, TensorIr>) -> ReduceKey {
    let reduce = &info.reduce;
    let input = tensors.get(&reduce.op.input.id).unwrap();
    let out = tensors.get(&reduce.op.out.id).unwrap();
    let problem = ReduceRoutineProblem::stated(
        &Geometry::row_major(&input.shape),
        &Geometry::row_major(&out.shape),
        reduce.axis,
        reduce_operation(reduce.inst),
        ReduceElements {
            input: dtype_to_storage_type(input.dtype),
            accumulator: reduce.acc.into_elem(),
            out: dtype_to_storage_type(out.dtype),
        },
    );
    ReduceKey::from(&problem)
}

/// The reduce on its elected routine, through the trace's read block, then the write block; on the
/// original inputs, the fallback where it does not run fused, since that run's result is the one
/// kept.
fn tune_fused(input: TuneInput<ReduceOptimizationTuneArg>) -> Result<TuneOutput, String> {
    let is_original = input.is_original();
    input.execute(|context, opt| match opt.execute_fused(context) {
        Ok(output) => Ok(output),
        Err(_) if is_original => Ok(opt.execute_fallback(context)),
        Err(error) => Err(format!("{error:?}")),
    })
}

/// Executes the fallback path for a reduction optimization.
fn tune_fallback(input: TuneInput<ReduceOptimizationTuneArg>) -> Result<TuneOutput, String> {
    input.execute(|ctx, opt| {
        opt.execute_fallback(ctx);
    });
    Ok(TuneOutput::UnChecked)
}
