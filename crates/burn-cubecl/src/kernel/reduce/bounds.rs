use burn_backend::cubecl::{autotune::with_roofline_bounds, dtype_to_storage_type};
use cubecl::{std::throughput::roofline_bounds, tune::TunableSet};
use cubek::reduce::{
    ReduceDtypes, components::instructions::ReduceOperationConfig, routines::ReduceCost,
};

use crate::{CubeAutotuneKey, tensor::CubeTensor};

/// Registers the performance bounds used for whole-tensor sum autotuning.
pub(super) fn with_sum_bounds<Out: 'static>(
    set: TunableSet<CubeAutotuneKey, CubeTensor, Out>,
) -> TunableSet<CubeAutotuneKey, CubeTensor, Out> {
    with_roofline_bounds(
        set,
        |_key: &CubeAutotuneKey, input: &CubeTensor, thresholds| {
            let elem = dtype_to_storage_type(input.dtype);

            // A whole-tensor sum is one fold, over every element, into a single output.
            let cost = ReduceCost {
                reduce_len: input.meta.num_elements(),
                reduce_count: 1,
                instruction: ReduceOperationConfig::Sum,
                dtypes: ReduceDtypes {
                    input: elem,
                    output: elem,
                    accumulation: elem,
                },
            };

            roofline_bounds(&input.client, cost.compute_key(), cost.work(), thresholds)
        },
    )
}
