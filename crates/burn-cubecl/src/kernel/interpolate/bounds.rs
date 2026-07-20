use alloc::sync::Arc;
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::ops::{InterpolateMode, InterpolateOptions};
use cubecl::{
    client::ComputeClient,
    std::throughput::measure_peak_throughput,
    throughput::{ThroughputKey, ThroughputMode},
    tune::{AutotuneBound, Bounds, BoundsGenerator, calculate_bounds},
};
use cubek::interpolate::launch::InterpolateAutotuneKey;

use crate::{CubeRuntime, tensor::CubeTensor};

type Inputs<R> = (CubeTensor<R>, [usize; 2], InterpolateOptions);
type BoundsGen<R> = dyn BoundsGenerator<InterpolateAutotuneKey, Inputs<R>> + Send + Sync;

const THRESHOLD: f32 = 1.0;

/// Creates a closure that calculates performance bounds for interpolate autotuning.
pub(super) fn create_bounds<R: CubeRuntime>(client: &ComputeClient<R>) -> Arc<BoundsGen<R>> {
    let owned_client = client.clone();

    Arc::new(
        move |key: &InterpolateAutotuneKey, inputs: &Inputs<R>| Bounds {
            bounds: autotune_bounds(&owned_client, key, inputs),
            launch_overhead: measure_peak_throughput(
                &owned_client,
                ThroughputKey {
                    mode: ThroughputMode::Launch,
                },
            )
            .duration,
        },
    )
}

fn autotune_bounds<R: CubeRuntime>(
    client: &ComputeClient<R>,
    _key: &InterpolateAutotuneKey,
    (input, output_size, options): &Inputs<R>,
) -> Vec<AutotuneBound> {
    let elem_input = dtype_to_elem_type(input.dtype);
    let elem_out = elem_input;

    let compute_throughput = measure_peak_throughput(
        client,
        ThroughputKey {
            mode: ThroughputMode::ComputeDirect { dtype: elem_out },
        },
    );

    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory,
    };
    let memory_throughput = measure_peak_throughput(client, memory_key);

    let input_bytes = input.meta.num_elements() * elem_input.size();

    let batch_size = input.meta.shape()[0];
    let channels = input.meta.shape()[1];
    let output_num_elements = batch_size * channels * output_size[0] * output_size[1];

    let out_bytes = output_num_elements * elem_out.size();
    let total_bytes = input_bytes + out_bytes;

    let ops_per_element = match options.mode {
        InterpolateMode::Nearest | InterpolateMode::NearestExact => 1, // Mostly index calculation
        InterpolateMode::Bilinear => 8,                                // 4 points * ~2 ops
        InterpolateMode::Bicubic => 40, // 16 points * ~2 ops + weight math
        InterpolateMode::Lanczos3 => 90, // 36 points * ~2 ops + sinc math
    };
    let total_ops = output_num_elements * ops_per_element;

    calculate_bounds(
        &compute_throughput,
        total_ops,
        THRESHOLD,
        &memory_throughput,
        total_bytes,
        THRESHOLD,
    )
}
