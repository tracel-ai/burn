use alloc::sync::Arc;
use burn_backend::cubecl::dtype_to_elem_type;
use cubecl::{
    client::ComputeClient,
    ir::ElemType,
    std::throughput::measure_peak_throughput,
    throughput::{ThroughputKey, ThroughputMode},
    tune::{AutotuneBound, Bounds, BoundsGenerator, calculate_bounds},
};
use cubek::reduce::{
    ReduceDtypes, ReduceWithIndicesDtypes, components::instructions::ReduceOperationConfig,
    launch::tune_key::ReduceAutotuneKey,
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

/// Fraction of the modeled roofline a reduce is expected to reach. Reduces are memory bound,
/// and a well-vectorized one streams close to peak bandwidth.
const THRESHOLD: f32 = 1.0;

/// Creates a closure that calculates performance bounds for reduce autotuning.
pub(super) fn create_reduce_bounds<R: CubeRuntime>() -> Arc<BoundsGen<ReduceAutotuneKey, Inputs<R>>>
{
    Arc::new(
        |_key: &ReduceAutotuneKey, (input, output, _axis, config, dtypes): &Inputs<R>| {
            bounds(
                &input.client,
                dtypes.accumulation.elem_type(),
                reduce_steps(input, output) * ops_per_reduce_step(config),
                tensor_bytes(input) + tensor_bytes(output),
            )
        },
    )
}

/// Creates a closure that calculates performance bounds for the fused top-k autotuning.
pub(super) fn create_reduce_with_indices_bounds<R: CubeRuntime>()
-> Arc<BoundsGen<ReduceAutotuneKey, InputsWithIndices<R>>> {
    Arc::new(
        |_key: &ReduceAutotuneKey,
         (input, values, indices, _axis, _k, dtypes): &InputsWithIndices<R>| {
            bounds(
                &input.client,
                dtypes.accumulation.elem_type(),
                // Top-k keeps a value and its index, so each step pays the compare twice.
                reduce_steps(input, values) * 2,
                tensor_bytes(input) + tensor_bytes(values) + tensor_bytes(indices),
            )
        },
    )
}

/// Creates a closure that calculates performance bounds for whole-tensor sum autotuning.
pub(super) fn create_sum_bounds<R: CubeRuntime>() -> Arc<BoundsGen<CubeAutotuneKey, CubeTensor<R>>>
{
    Arc::new(|_key: &CubeAutotuneKey, input: &CubeTensor<R>| {
        let elem = dtype_to_elem_type(input.dtype);

        bounds(
            &input.client,
            elem,
            // Folding N elements into one scalar takes N - 1 adds.
            input.meta.num_elements().saturating_sub(1),
            tensor_bytes(input) + elem.size(),
        )
    })
}

/// Compute and memory bounds for a reduce performing `compute_ops` and moving `bytes`.
fn bounds<R: CubeRuntime>(
    client: &ComputeClient<R>,
    elem_acc: ElemType,
    compute_ops: usize,
    bytes: usize,
) -> Bounds {
    Bounds {
        bounds: autotune_bounds(client, elem_acc, compute_ops, bytes),
        launch_overhead: measure_peak_throughput(
            client,
            ThroughputKey {
                mode: ThroughputMode::Launch,
            },
        )
        .duration_per_op(),
    }
}

fn autotune_bounds<R: CubeRuntime>(
    client: &ComputeClient<R>,
    elem_acc: ElemType,
    compute_ops: usize,
    bytes: usize,
) -> Vec<AutotuneBound> {
    let compute_throughput = measure_peak_throughput(
        client,
        ThroughputKey {
            mode: ThroughputMode::ComputeDirect { dtype: elem_acc },
        },
    );

    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory,
    };

    let memory_throughput = measure_peak_throughput(client, memory_key);

    calculate_bounds(
        &compute_throughput,
        compute_ops,
        THRESHOLD,
        &memory_throughput,
        &memory_key,
        bytes,
        THRESHOLD,
    )
}

/// Number of accumulate steps: every input element except the ones seeding an output.
fn reduce_steps<R: CubeRuntime>(input: &CubeTensor<R>, output: &CubeTensor<R>) -> usize {
    input
        .meta
        .num_elements()
        .saturating_sub(output.meta.num_elements())
}

fn tensor_bytes<R: CubeRuntime>(tensor: &CubeTensor<R>) -> usize {
    tensor.meta.num_elements() * dtype_to_elem_type(tensor.dtype).size()
}

/// Minimum ALU operations an accumulate step costs, as a lower bound on the compute time.
fn ops_per_reduce_step(config: &ReduceOperationConfig) -> usize {
    match config {
        // Single binary op (add, multiply, compare, bitwise or/and).
        ReduceOperationConfig::Sum
        | ReduceOperationConfig::Prod
        | ReduceOperationConfig::Mean
        | ReduceOperationConfig::Max
        | ReduceOperationConfig::Min
        | ReduceOperationConfig::MaxAbs
        | ReduceOperationConfig::Any
        | ReduceOperationConfig::All => 1,
        // Compare + conditional move of value and index. Top-k compares against the current
        // k-th and only rarely inserts, so its floor is the same two ops.
        ReduceOperationConfig::ArgMax
        | ReduceOperationConfig::ArgMin
        | ReduceOperationConfig::ArgTopK(_)
        | ReduceOperationConfig::TopK(_) => 2,
    }
}
