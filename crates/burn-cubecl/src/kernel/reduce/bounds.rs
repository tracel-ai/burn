use alloc::sync::Arc;
use burn_backend::cubecl::dtype_to_elem_type;
use cubecl::{
    client::ComputeClient,
    std::throughput::{measure_launch_overhead, measure_peak_throughput},
    throughput::{ThroughputKey, ThroughputMode},
    tune::{AutotuneBound, Bounds, BoundsGenerator, calculate_bounds},
};
use cubek::reduce::{
    ReduceDtypes, components::instructions::ReduceOperationConfig,
    launch::tune_key::ReduceAutotuneKey,
};

use crate::{CubeRuntime, tensor::CubeTensor};

type Inputs<R> = (
    CubeTensor<R>,
    CubeTensor<R>,
    usize,
    ReduceOperationConfig,
    ReduceDtypes,
);

type BoundsGen<R> = dyn BoundsGenerator<ReduceAutotuneKey, Inputs<R>, AutotuneBound> + Send + Sync;

const THRESHOLD: f32 = 1.0;

/// Creates a closure that calculates performance bounds for reduce autotuning.
pub(super) fn create_reduce_bounds<R: CubeRuntime>(client: &ComputeClient<R>) -> Arc<BoundsGen<R>> {
    let owned_client = client.clone();

    Arc::new(move |_key: &ReduceAutotuneKey, inputs: &Inputs<R>| Bounds {
        bounds: autotune_bounds(&owned_client, inputs),
        launch_overhead: measure_launch_overhead(&owned_client),
    })
}

fn autotune_bounds<R: CubeRuntime>(
    client: &ComputeClient<R>,
    (input, output, _axis, config, dtypes): &Inputs<R>,
) -> Vec<AutotuneBound> {
    let elem_input = dtype_to_elem_type(input.dtype);
    let elem_output = dtype_to_elem_type(output.dtype);
    let elem_acc = dtypes.accumulation.elem_type();

    let compute_throughput = measure_peak_throughput(
        client,
        ThroughputKey {
            mode: ThroughputMode::ComputeDirect,
            dtype: elem_acc,
        },
    );

    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory,
        dtype: elem_output,
    };

    let memory_throughput = measure_peak_throughput(client, memory_key);

    let num_input = input.meta.num_elements();
    let num_output = output.meta.num_elements();
    let input_bytes = num_input * elem_input.size();
    let output_bytes = num_output * elem_output.size();

    // Total ops = (N - C) scaled by ops per step.
    let compute_ops = (num_input - num_output) * ops_per_reduce_step(config);

    calculate_bounds(
        &compute_throughput,
        compute_ops,
        THRESHOLD,
        &memory_throughput,
        &memory_key,
        input_bytes + output_bytes,
        THRESHOLD,
    )
}

/// Number of ALU operations each input element contributes during reduction.
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
        // Compare + conditional move of both value and index.
        ReduceOperationConfig::ArgMax | ReduceOperationConfig::ArgMin => 2,
        // Sorted insertion into a k-element accumulator.
        ReduceOperationConfig::ArgTopK(k) | ReduceOperationConfig::TopK(k) => *k,
    }
}
