//! Attention's forward pass through cubek's autotune over its tile-DSL routines
//! ([`launch_tuned`]), for the attentions [`forward_serves`] admits.

use crate::{
    ops::{expand, numeric::full_device_dtype, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::ops::{AttentionModuleOptions, CausalAlignment};
use burn_backend::{DType, Shape, cubecl::dtype_to_storage_type};
use cubecl::features::TypeUsage;
use cubecl::prelude::InputScalar;
use cubek::attention::kernels::select::selector::QueryPositions;
use cubek::attention::kernels::tune::{AttentionTuneInputs, launch_tuned};
use cubek::attention::kernels::{
    AttentionScoring, AttentionTensors, AttentionVisibility, QueryPlacement, ScoreScale,
};
use cubek::std::autotune::GroupPriority;

/// Whether the tile routines take `query`, `key` and `value` as operands: one element, which they
/// are verified on and the device runs, and query heads that split evenly over the key heads.
pub(crate) fn routines_take(query: &CubeTensor, key: &CubeTensor, value: &CubeTensor) -> bool {
    let (q_shape, k_shape) = (query.meta.shape(), key.meta.shape());
    let element = dtype_to_storage_type(query.dtype);
    let elements = key.dtype == query.dtype
        && value.dtype == query.dtype
        && AttentionTensors::VERIFIED.contains(&element)
        && query
            .client
            .properties()
            .type_usage(element)
            .is_superset(TypeUsage::Arithmetic | TypeUsage::Buffer);
    let grouped = k_shape[1] != 0 && q_shape[1].is_multiple_of(k_shape[1]);
    elements && grouped
}

/// Whether the tile routines serve the forward pass of an attention of `query` against `key` and
/// `value` with `options`: operands they [take](routines_take), no additive bias and no softcap.
/// A mask, a custom scale and a causal mask anchored at either corner they read.
pub fn forward_serves(
    query: &CubeTensor,
    key: &CubeTensor,
    value: &CubeTensor,
    has_bias: bool,
    options: &AttentionModuleOptions,
) -> bool {
    routines_take(query, key, value) && !has_bias && options.softcap.is_none()
}

/// Executes attention through cubek's autotune over its tile-DSL routines.
///
/// # Panics
///
/// An attention [`forward_serves`] does not admit.
pub fn attention_autotune(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    mask: Option<CubeTensor>,
    attn_bias: Option<CubeTensor>,
    options: AttentionModuleOptions,
) -> CubeTensor {
    assert!(
        attn_bias.is_none() && options.softcap.is_none(),
        "the tile-DSL attention takes no additive bias or softcap"
    );
    let (client, device, dtype) = (query.client.clone(), query.device.clone(), query.dtype);
    let [batch, heads, seq_q, head_dim] = query.meta.shape().dims::<4>();
    let seq_kv = key.meta.shape()[2];
    let lengths = full_device_dtype(
        client.clone(),
        Shape::new([batch]),
        device.clone(),
        InputScalar::new(seq_kv as u32, dtype_to_storage_type(DType::U32)),
        DType::U32,
    );
    let scoring = AttentionScoring {
        visibility: AttentionVisibility::new(
            options.is_causal,
            None,
            match options.causal_alignment {
                CausalAlignment::BottomRight => QueryPlacement::Last,
                CausalAlignment::TopLeft => QueryPlacement::First,
            },
        ),
        scale: match options.scale {
            Some(scale) => ScoreScale::Custom(scale as f32),
            None => ScoreScale::standard(head_dim),
        },
    };
    let inputs = AttentionTuneInputs::new(
        &client,
        query.plain(),
        key.plain(),
        value.plain(),
        None,
        lengths.plain(),
        None,
        scoring,
    )
    .unwrap_or_else(|why| panic!("no tile-DSL attention routine serves these operands: {why:?}"))
    // burn's key and value are the call's own, every position attended: the autotune key holds
    // their extent rather than standing for a cache's.
    .attending_every_position();
    // A mask broadcast over the batch or the heads is read at stride zero there, never copied.
    let inputs = match mask {
        Some(mask) => {
            let scores = Shape::new([batch, heads, seq_q, seq_kv]);
            inputs.with_mask(expand(mask, scores).plain())
        }
        None => inputs,
    };
    let out = launch_tuned(
        QueryPositions::of(seq_q),
        &device,
        inputs,
        GroupPriority::DEFAULT,
    );
    // The routines write `[b, q, h, dv]`, a token's heads adjacent; burn's attention is
    // `[b, h, q, dv]`, viewed without a copy.
    let out = CubeTensor::new(client, out.handle, *out.metadata, device, dtype);
    swap_dims(out, 1, 2)
}
