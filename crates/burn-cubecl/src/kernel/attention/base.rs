#[cfg(feature = "autotune")]
use super::{attention_autotune, forward_serves};
use crate::{CubeBackend, tensor::CubeTensor};
use burn_backend::ops::{
    AttentionCache, AttentionModuleOptions,
    attention::{attention_fallback, attention_fallback_over_cache},
};

#[derive(Debug)]
/// Strategy used to select which attention implementation to run.
pub enum AttentionStrategy {
    /// Separate kernels: the scores materialized, then the softmax, then the product with the
    /// values.
    Fallback,

    /// cubek's tile-DSL routine measured fastest for the attention on this device, where the
    /// routines serve it ([`forward_serves`]), and the fallback otherwise.
    #[cfg(feature = "autotune")]
    Autotune,
}

impl Default for AttentionStrategy {
    fn default() -> Self {
        // if autotune is enabled, default to autotune
        #[cfg(feature = "autotune")]
        return AttentionStrategy::Autotune;

        // if autotune is disabled, default to fallback to make sure it runs
        #[cfg(not(feature = "autotune"))]
        AttentionStrategy::Fallback
    }
}

/// Launch an attention kernel with given strategy, over a key/value `cache` when one is given.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    mask: Option<CubeTensor>,
    attn_bias: Option<CubeTensor>,
    cache: Option<AttentionCache<CubeBackend>>,
    options: AttentionModuleOptions,
    strategy: AttentionStrategy,
) -> CubeTensor {
    match strategy {
        #[cfg(feature = "autotune")]
        AttentionStrategy::Autotune
            if forward_serves(
                &query,
                &key,
                &value,
                attn_bias.is_some(),
                mask.is_some() && cache.is_some(),
                &options,
            ) =>
        {
            attention_autotune(query, key, value, mask, cache, options)
        }
        _ => match cache {
            Some(cache) => attention_fallback_over_cache::<CubeBackend>(
                query, key, value, mask, attn_bias, cache, options,
            ),
            None => attention_fallback::<CubeBackend>(query, key, value, mask, attn_bias, options),
        },
    }
}
