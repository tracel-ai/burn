use alloc::sync::Arc;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    std::throughput::roofline_bounds,
    tune::{BoundsGenerator, Thresholds},
};
use cubek::reduce::{
    ReduceDtypes, ReduceWithIndicesDtypes, components::instructions::ReduceOperationConfig,
    launch::tune_key::ReduceAutotuneKey, routines::ReduceCost,
};

use crate::{CubeAutotuneKey, CubeRuntime, tensor::CubeTensor};

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
    usize,
    ReduceWithIndicesDtypes,
);

type BoundsGen<K, I> = dyn BoundsGenerator<K, I> + Send + Sync;

/// Fraction of the modeled roofline a reduce candidate is expected to reach.
const THRESHOLDS: Thresholds = Thresholds::uniform(1.0);

/// Creates a closure that calculates performance bounds for reduce autotuning.
pub(super) fn create_reduce_bounds<R: CubeRuntime>() -> Arc<BoundsGen<ReduceAutotuneKey, Inputs<R>>>
{
    Arc::new(
        |_key: &ReduceAutotuneKey, (input, _output, axis, instruction, dtypes): &Inputs<R>| {
            let cost = ReduceCost {
                reduce_len: input.meta.shape[*axis],
                reduce_count: folds(input, input.meta.shape[*axis]),
                instruction: *instruction,
                dtypes: *dtypes,
                indices: None,
            };

            roofline_bounds(&input.client, cost.compute_key(), cost.work(), THRESHOLDS)
        },
    )
}

/// Creates a closure that calculates performance bounds for the fused top-k autotuning.
pub(super) fn create_reduce_with_indices_bounds<R: CubeRuntime>()
-> Arc<BoundsGen<ReduceAutotuneKey, InputsWithIndices<R>>> {
    Arc::new(
        |_key: &ReduceAutotuneKey,
         (input, _values, indices, axis, k, dtypes): &InputsWithIndices<R>| {
            let cost = ReduceCost {
                reduce_len: input.meta.shape[*axis],
                reduce_count: folds(input, input.meta.shape[*axis]),
                instruction: ReduceOperationConfig::ArgTopK(*k),
                dtypes: ReduceDtypes {
                    input: dtypes.input,
                    output: dtypes.values,
                    accumulation: dtypes.accumulation,
                },
                indices: Some(dtype_to_storage_type(indices.dtype)),
            };

            roofline_bounds(&input.client, cost.compute_key(), cost.work(), THRESHOLDS)
        },
    )
}

/// Creates a closure that calculates performance bounds for whole-tensor sum autotuning.
pub(super) fn create_sum_bounds<R: CubeRuntime>() -> Arc<BoundsGen<CubeAutotuneKey, CubeTensor<R>>>
{
    Arc::new(|_key: &CubeAutotuneKey, input: &CubeTensor<R>| {
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
            indices: None,
        };

        roofline_bounds(&input.client, cost.compute_key(), cost.work(), THRESHOLDS)
    })
}

/// Number of independent folds: every axis of the input but the reduced one.
fn folds<R: CubeRuntime>(input: &CubeTensor<R>, reduce_len: usize) -> usize {
    input.meta.num_elements() / reduce_len.max(1)
}
