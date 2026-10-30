//! Attention's forward pass with each query's logsumexp, and its backward pass, through cubek's
//! tile-DSL routines: what autodiff records instead of the decomposed attention, for the
//! attentions [`backward_serves`] admits.

use crate::{
    CubeBackend, CubeDevice,
    ops::{numeric::full_device_dtype, swap_dims},
    tensor::CubeTensor,
};
use burn_backend::ops::{AttentionBackward, AttentionModuleOptions, CausalAlignment};
use burn_backend::{DType, Shape, cubecl::dtype_to_storage_type};
use cubecl::client::Client;
use cubecl::features::TypeUsage;
use cubecl::prelude::InputScalar;
use cubecl::std::tensor::TensorHandle;
use cubek::attention::kernels::select::selector::QueryPositions;
use cubek::attention::kernels::tune::{AttentionTuneInputs, launch_tuned};
use cubek::attention::kernels::{
    AttentionScoring, AttentionTensors, AttentionVisibility, QueryPlacement, ScoreScale,
};
use cubek::attention_backward::{AttentionBackwardTensors, attention_backward as launch_backward};
use cubek::matmul::kernels::tensor::PlainTensor;
use cubek::std::autotune::GroupPriority;

/// Whether cubek's forward pass with logsumexp and its backward pass serve an attention of
/// `query` against `key` and `value` with `options`, no mask and no additive bias: operands of one
/// element the routines take and the device runs, query heads that split evenly over the key
/// heads, no softcap, and a causal mask anchored at the bottom right, the one cubek reads, or one
/// over as many queries as keys, where both anchors agree.
pub fn backward_serves(
    query: &CubeTensor,
    key: &CubeTensor,
    value: &CubeTensor,
    options: &AttentionModuleOptions,
) -> bool {
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
    let anchored = !options.is_causal
        || options.causal_alignment == CausalAlignment::BottomRight
        || q_shape[2] == k_shape[2];
    elements && grouped && options.softcap.is_none() && anchored
}

/// The attention of `query` over `key` and `value`, `[b, h, q, dv]`, and each query's logsumexp,
/// `[b, h, q]` in `f32`, through whichever tile routine autotune measures fastest.
///
/// # Panics
///
/// Operands no tile routine serves whatever the device: an element they do not take, or query heads
/// that do not split evenly over the key heads.
pub fn attention_with_logsumexp(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    options: AttentionModuleOptions,
) -> (CubeTensor, CubeTensor) {
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
    let inputs = AttentionTuneInputs::new(
        &client,
        plain(query),
        plain(key),
        plain(value),
        None,
        plain(lengths),
        None,
        AttentionScoring {
            visibility: visibility(&options),
            scale: score_scale(&options, head_dim),
        },
    )
    .unwrap_or_else(|why| panic!("no tile-DSL attention routine serves these operands: {why:?}"))
    .with_logsumexp();
    let logsumexp = inputs.logsumexp().expect("`with_logsumexp` binds one");
    let out = launch_tuned(
        QueryPositions::of(seq_q),
        &device,
        inputs,
        GroupPriority::DEFAULT,
    );
    // The routines write `[b, q, h, dv]`, a token's heads adjacent; burn's attention is
    // `[b, h, q, dv]`, viewed without a copy.
    let out = cube_tensor(out, &client, &device, dtype);
    let logsumexp = cube_tensor(logsumexp, &client, &device, DType::F32);
    (swap_dims(out, 1, 2), logsumexp)
}

/// The gradients of [`attention_with_logsumexp`]'s attention against its query, key and value,
/// from `out_grad`, through whichever routine of cubek's backward pass autotune measures fastest.
///
/// # Panics
///
/// As [`attention_with_logsumexp`].
pub fn attention_backward(
    query: CubeTensor,
    key: CubeTensor,
    value: CubeTensor,
    out: CubeTensor,
    logsumexp: CubeTensor,
    out_grad: CubeTensor,
    options: AttentionModuleOptions,
) -> AttentionBackward<CubeBackend> {
    let (client, device, dtype) = (query.client.clone(), query.device.clone(), query.dtype);
    let head_dim = query.meta.shape()[3];
    let tensors = AttentionBackwardTensors::new(
        &client,
        plain(query),
        plain(key),
        plain(value),
        plain(out),
        plain(out_grad),
        plain(logsumexp),
        score_scale(&options, head_dim).value(),
        visibility(&options),
    )
    .unwrap_or_else(|why| {
        panic!("cubek's attention backward pass declines these operands: {why:?}")
    });
    let gradients = launch_backward(&device, tensors);
    AttentionBackward::new(
        cube_tensor(gradients.query, &client, &device, dtype),
        cube_tensor(gradients.key, &client, &device, dtype),
        cube_tensor(gradients.value, &client, &device, dtype),
    )
}

/// `handle`, which cubek wrote, as a tensor of `dtype` on `client`'s `device`.
fn cube_tensor(
    handle: TensorHandle,
    client: &Client,
    device: &CubeDevice,
    dtype: DType,
) -> CubeTensor {
    CubeTensor::new(
        client.clone(),
        handle.handle,
        *handle.metadata,
        device.clone(),
        dtype,
    )
}

/// `tensor` as cubek binds it, read where it lies.
fn plain(tensor: CubeTensor) -> PlainTensor {
    let elem = dtype_to_storage_type(tensor.dtype);
    PlainTensor::new(tensor.binding(), elem)
}

/// Which keys `options` let a query see, the causal diagonal placed by their alignment.
fn visibility(options: &AttentionModuleOptions) -> AttentionVisibility {
    AttentionVisibility::new(
        options.is_causal,
        None,
        match options.causal_alignment {
            CausalAlignment::BottomRight => QueryPlacement::Last,
            CausalAlignment::TopLeft => QueryPlacement::First,
        },
    )
}

/// The scale `options` give the scores, standard where they give none.
fn score_scale(options: &AttentionModuleOptions, head_dim: usize) -> ScoreScale {
    match options.scale {
        Some(scale) => ScoreScale::Custom(scale as f32),
        None => ScoreScale::standard(head_dim),
    }
}
