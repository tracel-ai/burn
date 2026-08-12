use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{std::throughput::roofline_bounds, tune::TunableSet};
use cubek::attention::forward::{
    definition::{AttentionCost, AttentionDims, AttentionGlobalTypes},
    launch::AttentionAutotuneKey,
};

use crate::{CubeRuntime, kernel::autotune_bounds, tensor::CubeTensor};

type Inputs<R> = (
    CubeTensor<R>,
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    Option<CubeTensor<R>>,
    burn_backend::ops::AttentionModuleOptions,
);

type AttentionTunables<R, Out> = TunableSet<AttentionAutotuneKey, Inputs<R>, Out>;

/// Registers the performance bounds used for attention autotuning.
pub(super) fn with_attention_bounds<R: CubeRuntime, Out: 'static>(
    set: AttentionTunables<R, Out>,
) -> AttentionTunables<R, Out> {
    autotune_bounds::with_bounds(set, |_key, inputs: &Inputs<R>, thresholds| {
        let client = &inputs.0.client;
        let cost = cost(inputs);

        roofline_bounds(client, cost.compute_key(client), cost.work(), thresholds)
    })
}

fn cost<R: CubeRuntime>(
    (query, key, value, mask, _attn_bias, options): &Inputs<R>,
) -> AttentionCost {
    let query_type = dtype_to_storage_type(query.dtype);

    AttentionCost {
        dims: AttentionDims {
            batch: query.meta.shape[0],
            num_heads: query.meta.shape[1],
            seq_q: query.meta.shape[2],
            head_dim: query.meta.shape[3],
            seq_kv: value.meta.shape[2],
            val_dim: value.meta.shape[3],
        },
        masked: mask.is_some(),
        causal: options.is_causal,
        types: AttentionGlobalTypes {
            query: query_type,
            key: dtype_to_storage_type(key.dtype),
            value: dtype_to_storage_type(value.dtype),
            // Stands in for an absent mask, which `masked: false` keeps out of the count.
            mask: mask
                .as_ref()
                .map(|mask| dtype_to_storage_type(mask.dtype))
                .unwrap_or(query_type),
            out: query_type,
        },
    }
}
