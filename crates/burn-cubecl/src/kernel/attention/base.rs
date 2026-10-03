#[cfg(feature = "autotune")]
use crate::kernel::attention::attention_autotune;
use crate::{CubeBackend, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{
    DType, Shape,
    ops::{AttentionModuleOptions, CausalAlignment, attention::attention_fallback},
};
use cubek::attention::forward::{
    definition::{
        AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError,
    },
    launch,
    routines::blackbox_accelerated::BlackboxAcceleratedStrategy,
};

#[derive(Debug)]
/// Strategy used to select which attention implementation to run.
pub enum AttentionStrategy {
    /// Flash Attention using accelerated inner matmuls.
    FlashBlackboxAccelerated(BlackboxAcceleratedStrategy),

    /// Flash Attention using unit inner matmuls.
    FlashUnit,

    /// Fallback implementation using multiple separate kernels.
    Fallback,

    /// Automatically benchmark and select the best strategy at runtime.
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

#[allow(clippy::too_many_arguments)]
/// Launch an attention kernel with given strategy
pub fn attention(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    mask: Option<CubeTensor>,
    attn_bias: Option<CubeTensor>,
    options: AttentionModuleOptions,
    strategy: AttentionStrategy,
) -> Result<CubeTensor, AttentionSetupError> {
    // Decided before autotune, so every candidate computes the same result.
    if flash_unsupported(&query, &key, &value, attn_bias.as_ref(), &options).is_some() {
        return Ok(attention_fallback::<CubeBackend>(
            query, key, value, mask, attn_bias, options,
        ));
    }

    // Resolve the flash launch strategy; the non-flash arms answer directly.
    let flash = match strategy {
        AttentionStrategy::FlashBlackboxAccelerated(strategy) => {
            launch::Strategy::BlackboxAccelerated(launch::BlueprintStrategy::Inferred(strategy))
        }
        AttentionStrategy::FlashUnit => {
            launch::Strategy::Unit(launch::BlueprintStrategy::Inferred(()))
        }
        AttentionStrategy::Fallback => {
            return Ok(attention_fallback::<CubeBackend>(
                query, key, value, mask, attn_bias, options,
            ));
        }
        #[cfg(feature = "autotune")]
        AttentionStrategy::Autotune => {
            return Ok(attention_autotune(
                query, key, value, mask, attn_bias, options,
            ));
        }
    };

    // The flash routines carry hard launch constraints — the accelerated
    // stage's `seq_q` must divide the problem's, and the tile head dim must
    // divide the problem's — while attention autotune caches its selection per
    // *anchored* key: a winner benchmarked on one raw `seq_q` (or `head_dim`)
    // must still run on the bucket's other raw shapes. So a flash strategy that
    // can't launch degrades to the fallback here, the same way the matmul
    // dispatch degrades a constrained routine to the unit kernel; the fallback
    // (separate kernels, materialized scores) has no such constraint and always
    // runs. Only an `InvalidConfig` degrades — availability and other errors
    // surface unchanged. `options` is `Copy`; the tensors are cheap handle
    // clones, taken only so the originals survive for the fallback.
    match flash_attention(
        query.clone(),
        key.clone(),
        value.clone(),
        mask.clone(),
        attn_bias.clone(),
        options,
        flash,
    ) {
        Err(AttentionSetupError::InvalidConfig(_)) => Ok(attention_fallback::<CubeBackend>(
            query, key, value, mask, attn_bias, options,
        )),
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
/// Launch a flash attention kernel
pub fn flash_attention(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    mask: Option<CubeTensor>,
    _attn_bias: Option<CubeTensor>,
    options: AttentionModuleOptions,
    strategy: launch::Strategy,
) -> Result<CubeTensor, AttentionSetupError> {
    if let Some(reason) = flash_unsupported(&query, &key, &value, _attn_bias.as_ref(), &options) {
        return Err(AttentionSetupError::InvalidConfig(Box::new(format!(
            "flash attention does not support {reason}"
        ))));
    }

    let client = query.client.clone();
    let out = init_attention_output(&query, &value);

    let dtypes = AttentionGlobalTypes {
        query: dtype_to_storage_type(query.dtype),
        key: dtype_to_storage_type(key.dtype),
        value: dtype_to_storage_type(value.dtype),
        mask: dtype_to_storage_type(mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8)),
        out: dtype_to_storage_type(out.dtype),
    };

    launch::launch_ref(
        strategy,
        &client,
        query.binding(),
        key.binding(),
        value.binding(),
        mask.map(|mask| mask.binding()),
        out.clone().binding(),
        &dtypes,
        AttentionOptions {
            causal: options.is_causal,
            accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::ElemType::Float(
                cubecl::ir::FloatKind::F32,
            )),
        },
    )?;

    Ok(out)
}

/// What the flash kernels can't compute in these options, if anything. They take no
/// scale, softcap or additive bias, and anchor the causal diagonal bottom-right.
pub(crate) fn flash_unsupported_options(
    options: &AttentionModuleOptions,
    has_bias: bool,
    seq_q: usize,
    seq_k: usize,
) -> Option<&'static str> {
    if has_bias {
        Some("an additive bias")
    } else if options.scale.is_some() {
        Some("a custom scale")
    } else if options.softcap.is_some() {
        Some("softcap")
    } else if options.is_causal
        && options.causal_alignment != CausalAlignment::BottomRight
        && seq_q != seq_k
    {
        Some("a causal mask that is not bottom-right aligned")
    } else {
        None
    }
}

/// What the flash kernels can't compute in this call, if anything. On top of
/// [`flash_unsupported_options`], they read K/V with the query's head index.
fn flash_unsupported(
    query: &CubeTensor,
    key: &CubeTensor,
    value: &CubeTensor,
    attn_bias: Option<&CubeTensor>,
    options: &AttentionModuleOptions,
) -> Option<&'static str> {
    let q_heads = query.meta.shape[1];
    if key.meta.shape[1] != q_heads || value.meta.shape[1] != q_heads {
        return Some("fewer key/value heads than query heads");
    }
    flash_unsupported_options(
        options,
        attn_bias.is_some(),
        query.meta.shape[2],
        key.meta.shape[2],
    )
}

pub(crate) fn init_attention_output(query: &CubeTensor, value: &CubeTensor) -> CubeTensor {
    let num_batches = query.meta.shape[0];
    let num_heads = query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let val_dim = value.meta.shape[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    empty_device_dtype(
        query.client.clone(),
        query.device.clone(),
        out_shape,
        query.dtype,
    )
}
