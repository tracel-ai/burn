//! Attention through cubek's autotunes over its tile-DSL routines alone
//! ([`launch_tuned`]): what the `tile-kernels-only` feature runs instead of the flash routines and
//! the fallback. An attention none of them serves fails rather than falling back, so a test run
//! lists what they don't serve yet.

use crate::{
    ops::{numeric::full_device_dtype, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::ops::AttentionModuleOptions;
use burn_backend::{DType, Shape, cubecl::dtype_to_storage_type};
use cubecl::prelude::InputScalar;
use cubek::attention::kernels::select::selector::QueryPositions;
use cubek::attention::kernels::tune::{AttentionTuneInputs, launch_tuned};
use cubek::attention::kernels::{AttentionScoring, AttentionVisibility, ScoreScale};
use cubek::matmul::kernels::tensor::PlainTensor;
use cubek::std::autotune::GroupPriority;

/// Executes attention through cubek's autotunes, over its tile-DSL routines alone.
///
/// # Panics
///
/// A mask tensor, an additive bias or a softcap, which the routines take none of, and operands
/// they serve whatever the device: an element they do not take, or query heads that do not split
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
        mask.is_none(),
        "the tile-DSL attention takes no mask tensor"
    );
    assert!(
        attn_bias.is_none() && options.softcap.is_none(),
        "the tile-DSL attention takes no additive bias or softcap"
    );
    let (client, device, dtype) = (query.client.clone(), query.device.clone(), query.dtype);
    let [batch, seq_q, head_dim] = [
        query.meta.shape()[0],
        query.meta.shape()[2],
        query.meta.shape()[3],
    ];
    let seq_kv = key.meta.shape()[2];
    let lengths = full_device_dtype(
        client.clone(),
        Shape::new([batch]),
        device.clone(),
        InputScalar::new(seq_kv as u32, dtype_to_storage_type(DType::U32)),
        DType::U32,
    );
    let scoring = AttentionScoring {
        visibility: AttentionVisibility::new(options.is_causal, None),
        scale: match options.scale {
            Some(scale) => ScoreScale::Custom(scale as f32),
            None => ScoreScale::standard(head_dim),
        },
    };
    let plain = |tensor: CubeTensor| {
        let elem = dtype_to_storage_type(tensor.dtype);
        PlainTensor::new(tensor.binding(), elem)
    };
    let inputs = AttentionTuneInputs::new(
        &client,
        plain(query),
        plain(key),
        plain(value),
        None,
        plain(lengths),
        None,
        scoring,
    )
    .unwrap_or_else(|why| panic!("no tile-DSL attention routine serves these operands: {why:?}"));
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
