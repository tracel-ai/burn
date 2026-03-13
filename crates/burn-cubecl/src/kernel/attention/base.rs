use crate::{
    CubeBackend, CubeRuntime, kernel::attention::attention_autotune,
    ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_backend::{
    DType, Shape,
    ops::{AttentionModuleOptions, attention::attention_fallback},
};
use cubek::attention::definition::{
    AccumulatorPrecision, AttentionGlobalTypes, AttentionOptions, AttentionSetupError,
};
use cubek::attention::launch;

#[derive(Debug)]
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

#[allow(clippy::too_many_arguments)]
/// Launch an attention kernel with given strategy
pub fn attention<R: CubeRuntime>(
    query: CubeTensor<R>,
    key: CubeTensor<R>,
    value: CubeTensor<R>,
    mask: Option<CubeTensor<R>>,
    attn_bias: Option<CubeTensor<R>>,
    options: AttentionModuleOptions,
    strategy: &AttentionStrategy,
    out: Option<CubeTensor<R>>,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let mut out = out.unwrap_or_else(|| init_attention_output(&query, &value));
    match strategy {
        AttentionStrategy::FlashBlackboxAccelerated => flash_attention(
            query,
            key,
            value,
            mask,
            attn_bias,
            options,
            out,
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
            out,
            launch::Strategy::Unit(cubek::attention::launch::BlueprintStrategy::Inferred(())),
        ),
        AttentionStrategy::Fallback => {
            out = attention_fallback::<CubeBackend<R, f32, i32, u8>>(
                query, key, value, mask, attn_bias, options,
            );
            Ok(out)
        }
        #[cfg(feature = "autotune")]
        AttentionStrategy::Autotune => {
            attention_autotune(query, key, value, mask, attn_bias, options, out)
        }
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
    out: CubeTensor<R>,
    strategy: launch::Strategy,
) -> Result<CubeTensor<R>, AttentionSetupError> {
    let client = query.client.clone();

    let dtypes = AttentionGlobalTypes {
        query: query.dtype.into(),
        key: key.dtype.into(),
        value: value.dtype.into(),
        mask: mask.as_ref().map(|m| m.dtype).unwrap_or(DType::U8).into(),
        out: out.dtype.into(),
    };

    cubek::attention::launch::launch_ref::<R>(
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
