use alloc::sync::Arc;
use burn_backend::cubecl::{dtype_to_elem_type, dtype_to_storage_type};
use cubecl::{
    client::ComputeClient,
    ir::{ElemType, FloatKind, StorageType},
    std::throughput::measure_peak_throughput,
    throughput::{ThroughputKey, ThroughputMode, compute_throughput_key, select_cmma_tile},
    tune::{AutotuneBound, Bounds, BoundsGenerator, calculate_bounds},
};
use cubek::attention::forward::launch::AttentionAutotuneKey;

use crate::{CubeRuntime, tensor::CubeTensor};

type Inputs<R> = (
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    Option<CubeTensor<R>>,
    burn_backend::ops::AttentionModuleOptions,
);

type BoundsGen<R> = dyn BoundsGenerator<AttentionAutotuneKey, Inputs<R>> + Send + Sync;

const THRESHOLD: f32 = 1.0;

/// Creates a closure that calculates performance bounds for attention autotuning.
pub(super) fn create_attention_bounds<R: CubeRuntime>(
    client: &ComputeClient<R>,
) -> Arc<BoundsGen<R>> {
    let owned_client = client.clone();

    Arc::new(
        move |_key: &AttentionAutotuneKey, inputs: &Inputs<R>| Bounds {
            bounds: autotune_bounds(&owned_client, inputs),
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
    (query, key, value, mask, _attn_bias, _options): &Inputs<R>,
) -> Vec<AutotuneBound> {
    let elem_q = dtype_to_storage_type(query.dtype);
    let elem_k = dtype_to_storage_type(key.dtype);

    let total_batches = query.meta.shape[0] * query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let head_dim = query.meta.shape[3];
    let seq_kv = value.meta.shape[2];
    let val_dim = value.meta.shape[3];

    // Estimate compute bounds with Q@K.T and S@V matmuls, using f32 accumulators.
    let acc_type = StorageType::Scalar(ElemType::Float(FloatKind::F32));

    let cmma_tile = select_cmma_tile(client, elem_q, elem_k, acc_type, (seq_q, seq_kv, head_dim));

    let compute_key = compute_throughput_key(
        cmma_tile,
        dtype_to_elem_type(query.dtype),
        ElemType::Float(FloatKind::F32),
    );

    let compute_throughput = measure_peak_throughput(client, compute_key);

    // Memory throughput bounds
    let memory_key = ThroughputKey {
        mode: ThroughputMode::Memory,
    };

    let memory_throughput = measure_peak_throughput(client, memory_key);

    let bytes = |t: &CubeTensor<R>| t.meta.num_elements() * dtype_to_elem_type(t.dtype).size();

    let mask_bytes = mask.as_ref().map(bytes).unwrap_or(0);
    // Output size: [total_batches, seq_q, val_dim]
    let out_bytes = total_batches * seq_q * val_dim * dtype_to_elem_type(query.dtype).size();

    let total_ops = total_batches * seq_q * seq_kv * (2 * head_dim - 1)
        + total_batches * seq_q * val_dim * (2 * seq_kv - 1);
    // Note: attn_bias is excluded as fast paths (flash attention) don't read it.
    let total_bytes = bytes(query) + bytes(key) + bytes(value) + mask_bytes + out_bytes;

    calculate_bounds(
        &compute_throughput,
        total_ops,
        THRESHOLD,
        &memory_throughput,
        total_bytes,
        THRESHOLD,
    )
}
