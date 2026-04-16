use core::f32;
#[allow(unused_imports)]
use num_traits::Float as _;

use burn_std::Shape;

use crate::{
    Backend, TensorMetadata, get_device_settings,
    ops::AttentionModuleOptions,
    tensor::{BoolTensor, FloatTensor},
};

/// Computes softmax(QKᵗ * scale) · V using separate kernels.
/// Serves as a fallback when FlashAttention is not used.
pub fn attention_fallback<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    mask: Option<BoolTensor<B>>,
    attn_bias: Option<FloatTensor<B>>,
    options: AttentionModuleOptions,
) -> FloatTensor<B> {
    if let Some(softcap) = options.softcap {
        assert!(softcap > 0.0, "softcap must be positive, got {softcap}");
    }

    // Attention scores: A = QKᵗ * scale
    let query_shape = query.shape().dims::<4>();
    let scale = options
        .scale
        .unwrap_or_else(|| 1.0 / (*query_shape.last().unwrap() as f64).sqrt());
    let transposed_key = B::float_transpose(key);
    let qk = B::float_matmul(query, transposed_key);
    let attention_scores = B::float_mul_scalar(qk, scale.into());

    // Softcap: softcap * tanh(scores / softcap)
    // Applied to raw logits before any -inf masking, so that tanh does not
    // map -inf to a finite value (which would break masking semantics).
    let attention_scores = if let Some(softcap) = options.softcap {
        let scaled = B::float_div_scalar(attention_scores, softcap.into());
        let tanh = B::float_tanh(scaled);
        B::float_mul_scalar(tanh, softcap.into())
    } else {
        attention_scores
    };

    // Bool masking
    let attention_scores = if let Some(mask) = mask {
        B::float_mask_fill(attention_scores, mask, f32::NEG_INFINITY.into())
    } else {
        attention_scores
    };

    // Causal masking: mask positions where col > row (future positions)
    let attention_scores = if options.is_causal {
        let causal_mask = build_causal_mask::<B>(&attention_scores);
        B::float_mask_fill(attention_scores, causal_mask, f32::NEG_INFINITY.into())
    } else {
        attention_scores
    };

    // Additive bias (ALiBi, relative position biases, etc.)
    let attention_scores = if let Some(bias) = attn_bias {
        B::float_add(attention_scores, bias)
    } else {
        attention_scores
    };

    // NaN-safe softmax: S = softmax(A)
    // When all positions in a row are masked (-inf), naive softmax has two NaN paths:
    //   (1) max is -inf, so the shift -inf - (-inf) = NaN;
    //   (2) after fixing (1), all exp values are 0, so sum is 0 and 0/0 = NaN.
    // Clamping max to finfo.min (most negative finite value) and sum to finfo.min_positive
    // (smallest positive normal) avoids both, yielding 0 for fully-masked rows.
    let finfo = attention_scores.dtype().finfo().expect("float tensor");
    let max_per_dim = B::float_max_dim(attention_scores.clone(), 3);
    let max_per_dim = B::float_clamp_min(max_per_dim, finfo.min.into());
    let minus_max = B::float_sub(attention_scores, max_per_dim);
    let numerator = B::float_exp(minus_max);
    let sum_exp = B::float_sum_dim(numerator.clone(), 3);
    let sum_exp = B::float_clamp_min(sum_exp, finfo.min_positive.into());
    let softmax = B::float_div(numerator, sum_exp);

    // Context: S · V
    B::float_matmul(softmax, value)
}

/// Builds a causal (upper-triangular) bool mask where `true` means "mask this position".
/// Shape: [batch_size, num_heads, seq_q, seq_k], masking positions where col > row.
fn build_causal_mask<B: Backend>(attention_scores: &FloatTensor<B>) -> BoolTensor<B> {
    let device = B::float_device(attention_scores);
    let scores_shape = attention_scores.shape().dims::<4>();
    let [batch_size, num_heads, seq_q, seq_k] = scores_shape;
    let settings = get_device_settings::<B>(&device);

    // row indices [seq_q, 1] and col indices [1, seq_k]
    // Offset col indices so that the causal boundary aligns at the bottom-right corner,
    // which handles cross-attention (seq_k > seq_q) correctly.
    let offset = seq_k as i64 - seq_q as i64;
    let rows = B::int_reshape(
        B::int_arange(0..seq_q as i64, &device, settings.int_dtype),
        Shape::new([seq_q, 1]),
    );
    let cols = B::int_reshape(
        B::int_arange(0..seq_k as i64, &device, settings.int_dtype),
        Shape::new([1, seq_k]),
    );

    // mask where col > row + offset (upper triangle)
    let rows_shifted = B::int_add_scalar(rows, offset.into());
    let mask_2d = B::int_lower(rows_shifted, cols, settings.bool_dtype);

    // Reshape to [1, 1, seq_q, seq_k] then expand to [batch_size, num_heads, seq_q, seq_k]
    let mask_4d = B::bool_reshape(mask_2d, Shape::new([1, 1, seq_q, seq_k]));
    B::bool_expand(mask_4d, Shape::new([batch_size, num_heads, seq_q, seq_k]))
}
