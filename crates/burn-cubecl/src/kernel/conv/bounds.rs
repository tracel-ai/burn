use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    client::ComputeClient,
    std::throughput::measure_peak_throughput,
    throughput::{ThroughputKey, ThroughputMode, compute_throughput_key, select_cmma_tile},
    tune::{AutotuneBound, calculate_bounds},
};

use crate::{CubeRuntime, kernel::conv::tune_key::ConvAutotuneKey};

const THRESHOLD: f32 = 1.0;

pub(crate) fn conv_autotune_bounds<R: CubeRuntime>(
    client: &ComputeClient<R>,
    key: &ConvAutotuneKey,
    input_num_elements: usize,
    weight_num_elements: usize,
    has_bias: bool,
) -> Vec<AutotuneBound> {
    let elem_input = dtype_to_storage_type(key.dtype);
    let elem_weight = dtype_to_storage_type(key.dtype);
    let elem_out = dtype_to_storage_type(key.dtype);

    let mut out_shape_prod = 1;
    for i in 0..key.shape.len() {
        let out_dim =
            (key.shape[i] + 2 * key.padding[i] - key.dilation[i] * (key.kernel_size[i] - 1) - 1)
                / key.stride[i]
                + 1;
        out_shape_prod *= out_dim;
    }

    // The total FLOPs (2·M·N·K per group) are invariant between forward, dgrad,
    // and wgrad because each is the adjoint of the others, so the same number of
    // FMAs is performed regardless of which pass is executing.
    let m = key.batch_size * out_shape_prod;
    let k = (key.in_channels / key.groups) * key.kernel_size.iter().product::<usize>();
    let n = key.out_channels / key.groups;

    let cmma_tile = select_cmma_tile(client, elem_input, elem_weight, elem_out, (m, n, k));

    let compute_key =
        compute_throughput_key(cmma_tile, elem_input.elem_type(), elem_out.elem_type());

    let compute_throughput = measure_peak_throughput(client, compute_key);

    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory,
    };
    let memory_throughput = measure_peak_throughput(client, memory_key);

    let output_num_elements = key.batch_size * key.out_channels * out_shape_prod;

    let input_bytes = input_num_elements * elem_input.elem_type().size();
    let weight_bytes = weight_num_elements * elem_weight.elem_type().size();
    let out_bytes = output_num_elements * elem_out.elem_type().size();
    let bias_bytes = if has_bias {
        key.out_channels * elem_out.elem_type().size()
    } else {
        0
    };

    let total_bytes = input_bytes + weight_bytes + out_bytes + bias_bytes;

    // Standard 2·M·N·K FMA count per group.
    let total_ops = key.groups * 2 * m * n * k;

    calculate_bounds(
        &compute_throughput,
        total_ops,
        THRESHOLD,
        &memory_throughput,
        total_bytes,
        THRESHOLD,
    )
}
