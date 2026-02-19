use crate::{CubeBackend, CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::{
    Backend, DType, Shape,
    ops::{AttentionModuleOptions, attention::attention_fallback},
};
use cubek::attention::definition::{
    AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError,
};
use cubek::attention::launch;

/// Strategy used to select which attention implementation to run.
pub enum AttentionStrategy {
    /// Flash Attention using accelerated inner matmuls.
    FlashBlackboxAccelerated,

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

/// Launch a flash attention kernel
pub fn attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    strategy: &AttentionStrategy,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    match strategy {
        AttentionStrategy::FlashBlackboxAccelerated => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            launch::Strategy::BlackboxAccelerated(
                cubek::attention::launch::BlueprintStrategy::Inferred(()),
            ),
        ),
        AttentionStrategy::FlashUnit => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            launch::Strategy::Unit(cubek::attention::launch::BlueprintStrategy::Inferred(())),
        ),
        AttentionStrategy::Fallback => Ok(attention_fallback::<CubeBackend<R, f32, i32, u8>>(
            query, key, value, mask, attn_bias, options,
        )),
        AttentionStrategy::Autotune => attention_autotune(),
    }
}

fn attention_autotune<R: CubeRuntime>() -> Result<CubeTensor<R>, AttentionSetupError> {
    todo!()
}

pub fn flash_attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    strategy: launch::Strategy,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let client = &query.client;
    let device = &query.device;

    let num_batches = query.meta.shape[0];
    let num_heads = query.meta.shape[1];
    let seq_q = query.meta.shape[2];
    let val_dim = value.meta.shape[3];
    let out_shape = Shape::new([num_batches, num_heads, seq_q, val_dim]);

    let out = empty_device_dtype::<R>(client.clone(), device.clone(), out_shape, query.dtype);
    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    cubek::attention::launch::launch_ref::<R>(
        strategy,
        client,
        &query.as_handle_ref(),
        &key.as_handle_ref(),
        &value.as_handle_ref(),
        &mask.as_ref().map(|mask| mask.as_handle_ref()),
        &out.as_handle_ref(),
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
