use crate::{
    CubeBackend, CubeRuntime, kernel::attention::attention_autotune,
    ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::{
    DType, Shape,
    ops::{AttentionModuleOptions, attention::attention_fallback},
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
pub fn attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    strategy: AttentionStrategy,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    // Resolve the flash launch strategy; the non-flash arms answer directly.
    let flash = match strategy {
        AttentionStrategy::FlashBlackboxAccelerated(strategy) => {
            launch::Strategy::BlackboxAccelerated(launch::BlueprintStrategy::Inferred(strategy))
        }
        AttentionStrategy::FlashUnit => {
            launch::Strategy::Unit(launch::BlueprintStrategy::Inferred(()))
        }
        AttentionStrategy::Fallback => {
            return Ok(attention_fallback::<CubeBackend<R>>(
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
        Err(AttentionSetupError::InvalidConfig(_)) => Ok(attention_fallback::<CubeBackend<R>>(
            query, key, value, mask, attn_bias, options,
        )),
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
/// Launch a flash attention kernel
pub fn flash_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    _attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    strategy: launch::Strategy,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let client = query.client.clone();
    let out = init_attention_output(&query, &value);

    let dtypes = AttentionGlobalTypes {
        query: dtype_to_storage_type(query.dtype),
        key: dtype_to_storage_type(key.dtype),
        value: dtype_to_storage_type(value.dtype),
        mask: dtype_to_storage_type(mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8)),
        out: dtype_to_storage_type(out.dtype),
    };

    launch::launch_ref::<R>(
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
            accumulator_precision: AccumulatorPrecision::Strict(cubecl::ir::StorageType::Scalar(
                cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F32),
            )),
        },
    )?;

    Ok(out)
}

pub(crate) fn init_attention_output<R: CubeRuntime>(
    query: &CubeTensor<R>,
    value: &CubeTensor<R>,
) -> CubeTensor<R> {
    let num_batches = query.meta.shape[0];
    let num_heads = query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let val_dim = value.meta.shape[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    empty_device_dtype::<R>(
        query.client.clone(),
        query.device.clone(),
        out_shape,
        query.dtype,
    )
}
