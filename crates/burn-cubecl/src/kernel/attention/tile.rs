//! Attention's forward pass through cubek's autotune over its tile-DSL routines
//! ([`launch_tuned`]), for the attentions [`forward_serves`] admits.

use core::num::NonZeroUsize;

use super::kept_arrivals;
use crate::{
    CubeBackend,
    kernel::cast,
    ops::{expand, numeric::full_device_dtype, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::ops::{AttentionCache, AttentionModuleOptions, CausalAlignment};
use burn_backend::{DType, Shape, cubecl::dtype_to_storage_type};
use cubecl::features::TypeUsage;
use cubecl::prelude::InputScalar;
use cubek::attention::kernels::select::problem::QueryPositions;
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

/// How `options` score an attention whose query heads are `head_dim` wide: the keys a query sees,
/// placed by their window and causal alignment, and the scale, standard where they give none.
pub(crate) fn scoring(options: &AttentionModuleOptions, head_dim: usize) -> AttentionScoring {
    AttentionScoring {
        visibility: visibility(options),
        scale: match options.scale {
            Some(scale) => ScoreScale::Custom(scale as f32),
            None => ScoreScale::standard(head_dim),
        },
    }
}

/// Which keys `options` let a query see. A window is bottom-right aligned and owns both edges,
/// while a causal edge without one sits where [`causal_alignment`](AttentionModuleOptions::causal_alignment)
/// places it.
pub(crate) fn visibility(options: &AttentionModuleOptions) -> AttentionVisibility {
    match options.sliding_window {
        Some(window) => AttentionVisibility::Windowed {
            window: NonZeroUsize::new(window as usize)
                .expect("attention sliding_window must be at least one key"),
            queries: QueryPlacement::Last,
        },
        None => AttentionVisibility::new(
            options.is_causal,
            None,
            match options.causal_alignment {
                CausalAlignment::BottomRight => QueryPlacement::Last,
                CausalAlignment::TopLeft => QueryPlacement::First,
            },
        ),
    }
}

/// Whether the tile routines can state `options`' visibility as one query placement: a window is
/// always bottom-right aligned, so a top-left causal edge over different query and key lengths
/// would need both edges at once, which a single placement cannot express.
pub(crate) fn visibility_serves(
    options: &AttentionModuleOptions,
    seq_q: usize,
    seq_k: usize,
) -> bool {
    options.sliding_window.is_none()
        || !options.is_causal
        || options.causal_alignment == CausalAlignment::BottomRight
        || seq_q == seq_k
}

/// Whether the tile routines serve the forward pass of an attention of `query` against `key` and
/// `value` with `options`: operands they [take](routines_take), no additive bias and no softcap.
/// A mask, a custom scale, a causal mask anchored at either corner and a sliding window they read,
/// except a top-left causal edge combined with a window over different query and key lengths, and
/// a mask over a key/value cache (`masked_cache`).
pub fn forward_serves(
    query: &CubeTensor,
    key: &CubeTensor,
    value: &CubeTensor,
    has_bias: bool,
    masked_cache: bool,
    options: &AttentionModuleOptions,
) -> bool {
    routines_take(query, key, value)
        && !has_bias
        && !masked_cache
        && options.softcap.is_none()
        && visibility_serves(options, query.meta.shape()[2], key.meta.shape()[2])
}

/// Executes attention through cubek's autotune over its tile-DSL routines, over a key/value
/// `cache` when one is given. Without one, every position of `key` and `value` is attended; with
/// one, each sequence attends its first `lengths[b]` positions, a single query among the routines
/// that serve one, counting in at counters kept for its device and stream.
///
/// # Panics
///
/// An attention [`forward_serves`] does not admit.
pub fn attention_autotune(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    mask: Option<CubeTensor>,
    cache: Option<AttentionCache<CubeBackend>>,
    options: AttentionModuleOptions,
) -> CubeTensor {
    assert!(
        options.softcap.is_none(),
        "the tile-DSL attention takes no softcap"
    );
    let (client, device, dtype) = (query.client.clone(), query.device.clone(), query.dtype);
    let [batch, heads, seq_q, head_dim] = query.meta.shape().dims::<4>();
    let seq_kv = key.meta.shape()[2];
    let positions = QueryPositions::of(seq_q);
    let scoring = scoring(&options, head_dim);
    let (query, key, value) = (query.plain(), key.plain(), value.plain());
    let inputs = match cache {
        Some(AttentionCache { lengths, pages }) => {
            assert!(
                mask.is_none(),
                "the tile-DSL attention takes no mask over a cache"
            );
            let arrivals = matches!(positions, QueryPositions::One)
                .then(|| kept_arrivals(&client, &device, batch * heads));
            AttentionTuneInputs::new(
                &client,
                query,
                key,
                value,
                pages.map(|pages| cast(pages, DType::U32).plain()),
                cast(lengths, DType::U32).plain(),
                arrivals.map(CubeTensor::plain),
                scoring,
            )
            .unwrap_or_else(|why| {
                panic!("no tile-DSL attention routine serves this cache: {why:?}")
            })
        }
        None => {
            let lengths = full_device_dtype(
                client.clone(),
                Shape::new([batch]),
                device.clone(),
                InputScalar::new(seq_kv as u32, dtype_to_storage_type(DType::U32)),
                DType::U32,
            );
            AttentionTuneInputs::new(
                &client,
                query,
                key,
                value,
                None,
                lengths.plain(),
                None,
                scoring,
            )
            .unwrap_or_else(|why| {
                panic!("no tile-DSL attention routine serves these operands: {why:?}")
            })
            // burn's key and value are the call's own, every position attended: the autotune key
            // holds their extent rather than standing for a cache's.
            .attending_every_position()
        }
    };
    // A mask broadcast over the batch or the heads is read at stride zero there, never copied.
    let inputs = match mask {
        Some(mask) => {
            let scores = Shape::new([batch, heads, seq_q, seq_kv]);
            inputs.with_mask(expand(mask, scores).plain())
        }
        None => inputs,
    };
    let out = launch_tuned(positions, &device, inputs, GroupPriority::DEFAULT);
    // The routines write `[b, q, h, dv]`, a token's heads adjacent; burn's attention is
    // `[b, h, q, dv]`, viewed without a copy.
    let out = CubeTensor::new(client, out.handle, *out.metadata, device, dtype);
    swap_dims(out, 1, 2)
}
