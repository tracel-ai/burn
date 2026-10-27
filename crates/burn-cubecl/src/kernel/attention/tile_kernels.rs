//! Attention through cubek's autotunes over its tile-DSL routines alone
//! ([`launch_tuned`]): what the `tile-kernels-only` feature runs instead of the flash routines and
//! the fallback. An attention none of them serves fails rather than falling back, so a test run
//! lists what they don't serve yet.

use crate::{
    ops::{expand, numeric::full_device_dtype, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::ops::{AttentionModuleOptions, CausalAlignment};
use burn_backend::{DType, Shape, cubecl::dtype_to_storage_type};
use cubecl::prelude::InputScalar;
use cubek::attention::kernels::select::selector::QueryPositions;
use cubek::attention::kernels::tune::{AttentionTuneInputs, launch_tuned};
use cubek::attention::kernels::{
    AttentionScoring, AttentionVisibility, QueryPlacement, ScoreScale,
};
use cubek::std::autotune::GroupPriority;

/// Executes attention through cubek's autotunes, over its tile-DSL routines alone.
///
/// # Panics
///
/// An additive bias or a softcap, which the routines take neither of, and operands they do not
/// serve whatever the device: an element they do not take, or query heads that do not split
/// evenly over the key heads.
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
    .unwrap_or_else(|why| panic!("no tile-DSL attention routine serves these operands: {why:?}"));
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
