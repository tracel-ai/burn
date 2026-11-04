#[cfg(feature = "autotune")]
use super::{attention_autotune, attention_over_cache_autotune, cache_serves, forward_serves};
use crate::{CubeBackend, tensor::CubeTensor};
use burn_backend::ops::{
    AttentionModuleOptions,
    attention::{attention_fallback, attention_over_cache_default},
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

/// Launch an attention kernel with given strategy
pub fn attention(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    mask: Option<CubeTensor>,
    attn_bias: Option<CubeTensor>,
    options: AttentionModuleOptions,
    strategy: AttentionStrategy,
) -> CubeTensor {
    match strategy {
        #[cfg(feature = "autotune")]
        AttentionStrategy::Autotune
            if forward_serves(&query, &key, &value, attn_bias.is_some(), &options) =>
        {
            attention_autotune(query, key, value, mask, attn_bias, options)
        }
        _ => attention_fallback::<CubeBackend>(query, key, value, mask, attn_bias, options),
    }
}

/// Launch an attention over a key/value cache: through cubek's autotune where its tile routines
/// serve the call ([`cache_serves`]), composed from tensor operations otherwise.
pub fn attention_over_cache(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    pages: Option<CubeTensor>,
    lengths: CubeTensor,
    options: AttentionModuleOptions,
) -> CubeTensor {
    #[cfg(feature = "autotune")]
    if cache_serves(&query, &key, &value, &options) {
        return attention_over_cache_autotune(query, key, value, pages, lengths, options);
    }
    attention_over_cache_default::<CubeBackend>(query, key, value, pages, lengths, options)
}
