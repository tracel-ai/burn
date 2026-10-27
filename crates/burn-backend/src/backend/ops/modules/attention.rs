#[allow(unused_imports)]
use num_traits::Float as _;

use burn_std::{FloatDType, Shape};

use crate::{
    Backend, TensorMetadata, get_or_init_device_settings,
    ops::{AttentionBackward, AttentionModuleOptions},
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

    /// What the scores are scaled by: `options`' scale, or `1 / √head_dim`.
    fn scale(&self, options: &AttentionModuleOptions) -> f64 {
        options
            .scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f64).sqrt())
    }

    /// `[batch, kv_heads, groups · seq_q, last]`: the query heads sharing a K/V head stacked as
    /// the rows of one matrix.
    fn grouped_rows(&self, last: usize) -> Shape {
        Shape::new([self.batch, self.kv_heads, self.groups() * self.seq_q, last])
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
    let shapes = AttentionShapes::new(&query.shape(), &key.shape(), &value.shape());
    let products = scaled_products::<B>(query, key, shapes.scale(&options), shapes);
    let scores = attention_scores::<B>(products, mask, attn_bias, shapes, options);
    let softmax = Softmax::<B>::new(scores);
    context::<B>(softmax.probabilities, value, shapes)
}

/// [`attention_fallback`] without a mask or an additive bias, and each query's logsumexp of its
/// scores, `[batch_size, num_heads, seq_len_q]` in `f32`: the forward
/// [`attention_with_logsumexp`](crate::ops::ModuleOps::attention_with_logsumexp) composes from
/// tensor operations. A query whose every key is masked holds a logsumexp no score reaches, so its
/// probabilities read back as zero.
pub fn attention_with_logsumexp_default<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    options: AttentionModuleOptions,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let shapes = AttentionShapes::new(&query.shape(), &key.shape(), &value.shape());
    let products = scaled_products::<B>(query, key, shapes.scale(&options), shapes);
    let scores = attention_scores::<B>(products, None, None, shapes, options);
    let softmax = Softmax::<B>::new(scores);
    let logsumexp = B::float_cast(
        B::float_reshape(
            B::float_add(softmax.max, B::float_log(softmax.sum)),
            Shape::new([shapes.batch, shapes.q_heads, shapes.seq_q]),
        ),
        FloatDType::F32,
    );
    (
        context::<B>(softmax.probabilities, value, shapes),
        logsumexp,
    )
}

/// The gradients of [`attention_with_logsumexp_default`]'s attention against its query, key and
/// value, from `out_grad`, the loss's gradient against the output, composed from tensor
/// operations: what [`attention_backward`](crate::ops::ModuleOps::attention_backward) runs where
/// a backend has no kernel of its own.
///
/// The probabilities are read back from `logsumexp`, and the gradient against the scores is
/// `p ⊙ (out_grad · valueᵀ − Σ out_grad ⊙ out)`, through the softcap's derivative where there is
/// one. The query heads sharing a key head are summed into its gradients.
pub fn attention_backward_default<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    value: FloatTensor<B>,
    out: FloatTensor<B>,
    logsumexp: FloatTensor<B>,
    out_grad: FloatTensor<B>,
    options: AttentionModuleOptions,
) -> AttentionBackward<B> {
    let shapes = AttentionShapes::new(&query.shape(), &key.shape(), &value.shape());
    let AttentionShapes {
        batch,
        q_heads,
        seq_q,
        seq_k,
        head_dim,
        val_dim,
        ..
    } = shapes;
    let scale = shapes.scale(&options);
    let products = scaled_products::<B>(query.clone(), key.clone(), scale, shapes);
    // `tanh(s / softcap)` of the scaled scores, what the softcap's derivative reads.
    let softcap_tanh = options
        .softcap
        .map(|softcap| B::float_tanh(B::float_div_scalar(products.clone(), softcap.into())));
    let scores = attention_scores::<B>(products, None, None, shapes, options);
    let logsumexp = B::float_cast(
        B::float_reshape(logsumexp, Shape::new([batch, q_heads, seq_q, 1])),
        scores.dtype().into(),
    );
    let probabilities = B::float_exp(B::float_sub(scores, logsumexp));
    let delta = B::float_sum_dim(B::float_mul(out_grad.clone(), out), 3);
    let out_grad = B::float_reshape(out_grad, shapes.grouped_rows(val_dim));
    let probabilities = B::float_reshape(probabilities, shapes.grouped_rows(seq_k));
    let value_grad = B::float_matmul(B::float_transpose(probabilities.clone()), out_grad.clone());
    let probabilities_grad = B::float_matmul(out_grad, B::float_transpose(value));
    let scores_grad = B::float_mul(
        probabilities,
        B::float_sub(
            probabilities_grad,
            B::float_reshape(delta, shapes.grouped_rows(1)),
        ),
    );
    // Through the softcap: d(c · tanh(s / c)) / ds = 1 − tanh²(s / c).
    let scores_grad = match softcap_tanh {
        Some(tanh) => {
            let squared = B::float_mul(tanh.clone(), tanh);
            let derivative = B::float_add_scalar(B::float_neg(squared), 1.0.into());
            B::float_mul(
                scores_grad,
                B::float_reshape(derivative, shapes.grouped_rows(seq_k)),
            )
        }
        None => scores_grad,
    };
    let scores_grad = B::float_mul_scalar(scores_grad, scale.into());
    let query_grad = B::float_reshape(
        B::float_matmul(scores_grad.clone(), key),
        Shape::new([batch, q_heads, seq_q, head_dim]),
    );
    let key_grad = B::float_matmul(
        B::float_transpose(scores_grad),
        B::float_reshape(query, shapes.grouped_rows(head_dim)),
    );
    AttentionBackward {
        query_grad,
        key_grad,
        value_grad,
    }
}

/// `query · keyᵗ · scale`, `[batch_size, num_heads, seq_len_q, seq_len_k]`.
///
/// Grouped-query attention: query head `h` reads K/V head `h / groups`. The query heads sharing a
/// K/V head are adjacent, so stacking them as rows ([`AttentionShapes::grouped_rows`]) turns GQA
/// into plain attention over `kv_heads` heads without repeating K and V. The products are laid
/// back as `[batch, q_heads, seq_q, seq_k]` so masks, bias and causality see query rows.
fn scaled_products<B: Backend>(
    query: FloatTensor<B>,
    key: FloatTensor<B>,
    scale: f64,
    shapes: AttentionShapes,
) -> FloatTensor<B> {
    let AttentionShapes {
        batch,
        q_heads,
        seq_q,
        seq_k,
        head_dim,
        ..
    } = shapes;
    let groups = shapes.groups();
    let query = if groups > 1 {
        B::float_reshape(query, shapes.grouped_rows(head_dim))
    } else {
        query
    };
    let transposed_key = B::float_transpose(key);
    let qk = B::float_matmul(query, transposed_key);
    let qk = if groups > 1 {
        B::float_reshape(qk, Shape::new([batch, q_heads, seq_q, seq_k]))
    } else {
        qk
    };
    B::float_mul_scalar(qk, scale.into())
}

/// The attention scores from the scaled `products`: capped, masked and biased, in the order
/// documented on [`attention`](crate::ops::ModuleOps::attention).
fn attention_scores<B: Backend>(
    products: FloatTensor<B>,
    mask: Option<BoolTensor<B>>,
    attn_bias: Option<FloatTensor<B>>,
    shapes: AttentionShapes,
    options: AttentionModuleOptions,
) -> FloatTensor<B> {
    // Softcap: softcap * tanh(scores / softcap)
    // Applied to raw logits before any -inf masking, so that tanh does not
    // map -inf to a finite value (which would break masking semantics).
    let attention_scores = if let Some(softcap) = options.softcap {
        assert!(softcap > 0.0, "softcap must be positive, got {softcap}");
        let scaled = B::float_div_scalar(products, softcap.into());
        let tanh = B::float_tanh(scaled);
        B::float_mul_scalar(tanh, softcap.into())
    } else {
        products
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
            options.causal_alignment.offset(shapes.seq_q, shapes.seq_k),
        );
        B::float_mask_fill(attention_scores, causal_mask, f32::NEG_INFINITY.into())
    } else {
        attention_scores
    };

    // Additive bias (ALiBi, relative position biases, etc.)
    if let Some(bias) = attn_bias {
        B::float_add(attention_scores, bias)
    } else {
        attention_scores
    }
}

/// The softmax of attention scores along their keys, with the max and the sum it divides by:
/// `logsumexp = max + log(sum)`.
struct Softmax<B: Backend> {
    probabilities: FloatTensor<B>,
    /// `[batch_size, num_heads, seq_len_q, 1]`.
    max: FloatTensor<B>,
    /// `[batch_size, num_heads, seq_len_q, 1]`.
    sum: FloatTensor<B>,
}

impl<B: Backend> Softmax<B> {
    /// NaN-safe softmax: S = softmax(A)
    ///
    /// When all positions in a row are masked (-inf), naive softmax has two NaN paths:
    ///   (1) max is -inf, so the shift -inf - (-inf) = NaN;
    ///   (2) after fixing (1), all exp values are 0, so sum is 0 and 0/0 = NaN.
    ///
    /// Clamping max to finfo.min (most negative finite value) and sum to finfo.min_positive
    /// (smallest positive normal) avoids both, yielding 0 for fully-masked rows.
    fn new(attention_scores: FloatTensor<B>) -> Self {
        let finfo = attention_scores.dtype().finfo().expect("float tensor");
        let max_per_dim = B::float_max_dim(attention_scores.clone(), 3);
        let max_per_dim = B::float_clamp_min(max_per_dim, finfo.min.into());
        let minus_max = B::float_sub(attention_scores, max_per_dim.clone());
        let numerator = B::float_exp(minus_max);
        let sum_exp = B::float_sum_dim(numerator.clone(), 3);
        let sum_exp = B::float_clamp_min(sum_exp, finfo.min_positive.into());
        Self {
            probabilities: B::float_div(numerator, sum_exp.clone()),
            max: max_per_dim,
            sum: sum_exp,
        }
    }
}

/// Context: S · V, `[batch_size, num_heads, seq_len_q, val_dim]`, with the query heads sharing a
/// key/value head stacked as rows as in [`scaled_products`].
fn context<B: Backend>(
    probabilities: FloatTensor<B>,
    value: FloatTensor<B>,
    shapes: AttentionShapes,
) -> FloatTensor<B> {
    if shapes.groups() > 1 {
        let probabilities = B::float_reshape(probabilities, shapes.grouped_rows(shapes.seq_k));
        let context = B::float_matmul(probabilities, value);
        B::float_reshape(
            context,
            Shape::new([shapes.batch, shapes.q_heads, shapes.seq_q, shapes.val_dim]),
        )
    } else {
        B::float_matmul(probabilities, value)
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
