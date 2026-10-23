#[allow(unused_imports)]
use num_traits::Float as _;

use burn_std::Shape;

use crate::{
    Backend, TensorMetadata, get_or_init_device_settings,
    ops::AttentionModuleOptions,
    tensor::{BoolTensor, FloatTensor},
};

/// Shapes of an attention call, validated against the [`attention`](crate::ops::ModuleOps::attention)
/// contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionShapes {
    /// Batch size.
    pub batch: usize,
    /// Number of query heads.
    pub q_heads: usize,
    /// Number of key/value heads. Divides `q_heads`.
    pub kv_heads: usize,
    /// Query sequence length.
    pub seq_q: usize,
    /// Key/value sequence length.
    pub seq_k: usize,
    /// Query/key head dimension.
    pub head_dim: usize,
    /// Value head dimension.
    pub val_dim: usize,
}

impl AttentionShapes {
    /// Query heads sharing one K/V head (`1` for multi-head attention).
    pub fn groups(&self) -> usize {
        self.q_heads / self.kv_heads
    }

    /// Validates the shapes of an attention call, panicking with a descriptive message
    /// when they do not satisfy the contract.
    ///
    /// Backends index K/V with these shapes (some through unchecked pointer arithmetic),
    /// so this must run before any kernel sees the tensors.
    pub fn new(query: &Shape, key: &Shape, value: &Shape) -> Self {
        assert!(
            query.num_dims() == 4 && key.num_dims() == 4 && value.num_dims() == 4,
            "attention: query, key and value must be 4D, got {query:?}, {key:?}, {value:?}"
        );
        let [batch, q_heads, seq_q, head_dim] = query.dims::<4>();
        let [k_batch, kv_heads, seq_k, k_dim] = key.dims::<4>();
        let [v_batch, v_heads, v_seq, val_dim] = value.dims::<4>();
        assert!(
            k_batch == batch && v_batch == batch,
            "attention: batch mismatch (query {batch}, key {k_batch}, value {v_batch})"
        );
        assert_eq!(k_dim, head_dim, "attention: key head_dim mismatch");
        assert!(head_dim > 0, "attention: head_dim must be non-zero");
        assert_eq!(
            v_heads, kv_heads,
            "attention: key and value head counts differ"
        );
        assert_eq!(
            v_seq, seq_k,
            "attention: key and value sequence lengths differ"
        );
        assert!(
            kv_heads > 0 && q_heads.is_multiple_of(kv_heads),
            "attention: query heads ({q_heads}) must be a multiple of key/value heads ({kv_heads})"
        );
        Self {
            batch,
            q_heads,
            kv_heads,
            seq_q,
            seq_k,
            head_dim,
            val_dim,
        }
    }
}

/// Computes softmax(QKᵗ * scale) · V using separate kernels.
/// Serves as a fallback when FlashAttention is not used.
///
/// Follows the order documented on [`attention`](crate::ops::ModuleOps::attention):
/// scale, softcap, bool and causal masks, additive bias, softmax.
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
    let shapes = AttentionShapes::new(&query.shape(), &key.shape(), &value.shape());
    let AttentionShapes {
        batch,
        q_heads,
        kv_heads,
        seq_q,
        seq_k,
        head_dim,
        val_dim,
    } = shapes;
    let groups = shapes.groups();

    // Grouped-query attention: query head `h` reads K/V head `h / groups`. The query
    // heads sharing a K/V head are adjacent, so folding them into the row dimension
    // (`[batch, kv_heads, groups * seq_q, head_dim]`) turns GQA into plain attention
    // over `kv_heads` heads without repeating K and V. Scores are unfolded back to
    // `[batch, q_heads, seq_q, seq_k]` so masks, bias and causality see query rows.
    let query = if groups > 1 {
        B::float_reshape(
            query,
            Shape::new([batch, kv_heads, groups * seq_q, head_dim]),
        )
    } else {
        query
    };

    // Attention scores: A = QKᵗ * scale
    let scale = options
        .scale
        .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());
    let transposed_key = B::float_transpose(key);
    let qk = B::float_matmul(query, transposed_key);
    let qk = if groups > 1 {
        B::float_reshape(qk, Shape::new([batch, q_heads, seq_q, seq_k]))
    } else {
        qk
    };
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
        let causal_mask = build_causal_mask::<B>(
            &attention_scores,
            options.causal_alignment.offset(seq_q, seq_k),
        );
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
    if groups > 1 {
        let softmax = B::float_reshape(
            softmax,
            Shape::new([batch, kv_heads, groups * seq_q, seq_k]),
        );
        let context = B::float_matmul(softmax, value);
        B::float_reshape(context, Shape::new([batch, q_heads, seq_q, val_dim]))
    } else {
        B::float_matmul(softmax, value)
    }
}

/// Builds a causal (upper-triangular) bool mask where `true` means "mask this position".
/// Shape: [batch_size, num_heads, seq_q, seq_k], masking positions where col > row + offset
/// (see [`CausalAlignment::offset`](crate::ops::CausalAlignment::offset)).
fn build_causal_mask<B: Backend>(attention_scores: &FloatTensor<B>, offset: i64) -> BoolTensor<B> {
    let device = attention_scores.device();
    let scores_shape = attention_scores.shape().dims::<4>();
    let [batch_size, num_heads, seq_q, seq_k] = scores_shape;
    let settings = get_or_init_device_settings::<B>(&device);

    // row indices [seq_q, 1] and col indices [1, seq_k]
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
