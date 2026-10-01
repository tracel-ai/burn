use super::{
    optimization::{ReduceOptimizationTuneArg, reduce_instruction2config},
    tune::FusedReduceAutotuneKey,
};
use crate::{engine::trace::TuneOutput, tune::FusionTuneInputs};
use burn_backend::cubecl::{autotune::with_roofline_bounds, dtype_to_storage_type};
use cubecl::{
    std::throughput::roofline_bounds,
    tune::{TunableSet, Work},
};
use cubek::reduce::{ReduceDtypes, routines::ReduceCost};

type Inputs = FusionTuneInputs<ReduceOptimizationTuneArg>;

/// Registers the roofline of the reduction together with the work fused around it: the fold's
/// arithmetic, and the traffic of the whole trace.
///
/// A round that meets it ends before the unfused fallback, whose own tune it would otherwise run.
pub(crate) fn with_fused_reduce_bounds(
    set: TunableSet<FusedReduceAutotuneKey, Inputs, TuneOutput>,
) -> TunableSet<FusedReduceAutotuneKey, Inputs, TuneOutput> {
    with_roofline_bounds(set, |_key, inputs, thresholds| {
        let info = &inputs.optimization().info;
        let tensors = inputs.tensors();
        let reduce = &info.reduce;
        let input = tensors
            .get(&reduce.op.input.id)
            .expect("reduce input registered in the context");
        let reduce_len = input.shape[reduce.axis];
        let cost = ReduceCost {
            reduce_len,
            reduce_count: input.shape.num_elements() / reduce_len.max(1),
            instruction: reduce_instruction2config(&reduce.inst),
            dtypes: ReduceDtypes {
                input: dtype_to_storage_type(reduce.op.input.dtype),
                output: dtype_to_storage_type(reduce.op.out.dtype),
                accumulation: reduce.acc.into_elem(),
            },
        };
        let work = Work {
            compute_ops: cost.compute_ops(),
            bytes: info.trace.traffic(tensors),
        };

        roofline_bounds(&info.client, cost.compute_key(), work, thresholds)
    })
}
