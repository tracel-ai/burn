use super::tune::ReduceDimAutotuneKey;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{std::throughput::roofline_bounds, tune::TunableSet};
use cubek::reduce::{
    ReduceDtypes, ReduceWithIndicesDtypes, components::instructions::ReduceOperationConfig,
    routines::ReduceCost,
};

use crate::{CubeAutotuneKey, CubeRuntime, kernel::autotune_bounds, tensor::CubeTensor};

type Inputs<R> = (
    CubeTensor<R>,
    CubeTensor<R>,
    usize,
    ReduceOperationConfig,
    ReduceDtypes,
);

type InputsWithIndices<R> = (
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
    usize,
    ReduceOperationConfig,
    ReduceWithIndicesDtypes,
);

/// Registers the performance bounds used for reduce autotuning.
pub(super) fn with_reduce_bounds<R: CubeRuntime, Out: 'static>(
    set: TunableSet<ReduceDimAutotuneKey, Inputs<R>, Out>,
) -> TunableSet<ReduceDimAutotuneKey, Inputs<R>, Out> {
    autotune_bounds::with_bounds(
        set,
        |_key, (input, _output, axis, instruction, dtypes): &Inputs<R>, thresholds| {
            let cost = ReduceCost {
                reduce_len: input.meta.shape[*axis],
                reduce_count: folds(input, input.meta.shape[*axis]),
                instruction: *instruction,
                dtypes: *dtypes,
            };

            roofline_bounds(&input.client, cost.compute_key(), cost.work(), thresholds)
        },
    )
}

/// Registers the performance bounds used for the fused top-k autotuning.
pub(super) fn with_reduce_with_indices_bounds<R: CubeRuntime, Out: 'static>(
    set: TunableSet<ReduceDimAutotuneKey, InputsWithIndices<R>, Out>,
) -> TunableSet<ReduceDimAutotuneKey, InputsWithIndices<R>, Out> {
    autotune_bounds::with_bounds(
        set,
        |_key,
         (input, _values, _indices, axis, config, dtypes): &InputsWithIndices<R>,
         thresholds| {
            let cost = ReduceCost {
                reduce_len: input.meta.shape[*axis],
                reduce_count: folds(input, input.meta.shape[*axis]),
                instruction: *config,
                dtypes: ReduceDtypes {
                    input: dtypes.input,
                    output: dtypes.values,
                    accumulation: dtypes.accumulation,
                },
            };

            roofline_bounds(&input.client, cost.compute_key(), cost.work(), thresholds)
        },
    )
}

/// Registers the performance bounds used for whole-tensor sum autotuning.
pub(super) fn with_sum_bounds<R: CubeRuntime, Out: 'static>(
    set: TunableSet<CubeAutotuneKey, CubeTensor<R>, Out>,
) -> TunableSet<CubeAutotuneKey, CubeTensor<R>, Out> {
    autotune_bounds::with_bounds(
        set,
        |_key: &CubeAutotuneKey, input: &CubeTensor<R>, thresholds| {
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

/// Number of independent folds: every axis of the input but the reduced one.
fn folds<R: CubeRuntime>(input: &CubeTensor<R>, reduce_len: usize) -> usize {
    input.meta.num_elements() / reduce_len.max(1)
}
