use core::f32;
#[allow(unused_imports)]
use num_traits::Float as _;

use crate::{
    ElementConversion, TensorMetadata,
    backend::Backend,
    ops::{BoolTensor, FloatTensor},
};

/// Computes softmax(QKᵗ / √d) · V using separate kernels.
/// Serves as a fallback when FlashAttention is not used.
pub fn naive_attention<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    mask: Option<BoolTensor<B>>,
) -> FloatTensor<B> {
    // Attention scores: A = QKᵗ / √d
    let query_shape = query.shape().dims::<4>();
    let sqrt_d = (*query_shape.last().unwrap() as f32).sqrt().elem();
    let transposed_key = B::float_transpose(key);
    let qk = B::float_matmul(query, transposed_key);
    let attention_scores = B::float_div_scalar(qk, sqrt_d);

    // Masking
    let attention_scores = if let Some(mask) = mask {
        B::float_mask_fill(attention_scores, mask, f32::NEG_INFINITY.elem())
    } else {
        attention_scores
    };

    // Softmax: S = softmax(A)
    let max_per_dim = B::float_max_dim(attention_scores.clone(), 3);
    let minus_max = B::float_sub(attention_scores, max_per_dim);
    let numerator = B::float_exp(minus_max);
    let sum_exp = B::float_sum_dim(numerator.clone(), 3);
    let softmax = B::float_div(numerator, sum_exp);

    // Context: S · V
    B::float_matmul(softmax, value)
}
