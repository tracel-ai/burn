use crate::backend::Backend;
use crate::ops::FloatTensor;
use crate::{ElementConversion, TensorMetadata};

/// Attention forward pass
///
/// Computes: softmax((Q K^T) / sqrt(d_k)) V
///
/// # Arguments
/// - `query` (Q): The query tensor of shape (batch, num_heads, seq_len_q, d_k).
/// - `key` (K): The key tensor of shape (batch, num_heads, seq_len_k, d_k).
/// - `value` (V): The value tensor of shape (batch, num_heads, seq_len_k, d_v).
/// - `mask_pad`: [TODO] Optional padding mask of shape (batch, 1, 1, seq_len_k).
/// - `mask_attn`: [TODO] Optional attention mask of shape (batch, num_heads, seq_len_q, seq_len_k).
///
/// # Returns
/// - A tensor of shape (batch, num_heads, seq_len_q, d_v) containing the attention output.
pub fn attention<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
) -> FloatTensor<B> {
    let scores = attention_scores::<B>(query, key);
    let weights = attention_weights::<B>(scores);
    B::float_matmul(weights, value)
}

/// Computes: (Q K^T) / sqrt(d_k)
fn attention_scores<B: Backend>(query: FloatTensor<B>, key: FloatTensor<B>) -> FloatTensor<B> {
    let d_k_sqrt = (query.shape().dims[3] as f32).sqrt();
    let scores = B::float_matmul(query, B::float_transpose(key));

    // TODO have dropout option?
    B::float_div_scalar(scores, B::FloatElem::from_elem(d_k_sqrt))
}

/// softmax(attn_scores) V
fn attention_weights<B: Backend>(attn_scores: FloatTensor<B>) -> FloatTensor<B> {
    // TODO include mask_pad?
    // TODO include mask_attn?
    // TODO use quiet softmax?

    B::softmax(attn_scores, 3)
}
