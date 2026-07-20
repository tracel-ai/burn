use burn_core as burn;

use burn::tensor::activation::softmax;
use burn::tensor::{Bool, Tensor};

/// Functional scaled dot-product attention, mirroring
/// [`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
///
/// Computes `softmax(Q Kᵀ * scale + mask) V` over caller-supplied query/key/value
/// tensors, without going through a packaged attention module. Convenient for models
/// that build their own q/k/v (e.g. RoPE-rotated or bias-added attention) and call
/// attention functionally.
///
/// # Shapes
///
/// - query: `[..., seq_len_q, d_k]`
/// - key:   `[..., seq_len_kv, d_k]`
/// - value: `[..., seq_len_kv, d_v]`
/// - output: `[..., seq_len_q, d_v]`
///
/// # Arguments
///
/// * `attn_mask`: optional additive mask broadcastable to the scores
///   `[..., seq_len_q, seq_len_kv]`; added before the softmax, so use a large negative
///   value (e.g. `f32::NEG_INFINITY`) to prevent attention. Cannot be combined with
///   `is_causal`.
/// * `is_causal`: if `true`, apply a causal mask so each query only attends to keys at
///   the same or earlier positions.
/// * `scale`: scale applied to the scores; defaults to `1 / sqrt(d_k)`.
pub fn scaled_dot_product_attention<const D: usize>(
    query: Tensor<D>,
    key: Tensor<D>,
    value: Tensor<D>,
    attn_mask: Option<Tensor<D>>,
    is_causal: bool,
    scale: Option<f64>,
) -> Tensor<D> {
    let d_k = query.dims()[D - 1];
    let scale = scale.unwrap_or(1.0 / (d_k as f64).sqrt());

    // [..., seq_len_q, seq_len_kv]
    let scores = query.matmul(key.transpose()).mul_scalar(scale);

    let scores = if is_causal {
        assert!(
            attn_mask.is_none(),
            "scaled_dot_product_attention: `is_causal` and `attn_mask` are mutually exclusive"
        );
        let dims = scores.dims();
        let (seq_q, seq_kv) = (dims[D - 2], dims[D - 1]);
        let mask = Tensor::<2, Bool>::triu_mask([seq_q, seq_kv], 1, &scores.device());
        scores.mask_fill(mask.unsqueeze::<D>(), f32::NEG_INFINITY)
    } else if let Some(mask) = attn_mask {
        scores.add(mask)
    } else {
        scores
    };

    softmax(scores, D - 1).matmul(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{TensorData, Tolerance};
    type FT = f32;

    fn qkv() -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let device = Default::default();
        let q = Tensor::<3>::from_data(
            TensorData::from([[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]]),
            &device,
        );
        let k = Tensor::<3>::from_data(
            TensorData::from([[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]]),
            &device,
        );
        let v = Tensor::<3>::from_data(TensorData::from([[[1.0, 2.0], [3.0, 4.0]]]), &device);
        (q, k, v)
    }

    // References from torch.nn.functional.scaled_dot_product_attention.
    #[test]
    fn sdpa_no_mask() {
        let (q, k, v) = qkv();
        let out = scaled_dot_product_attention(q, k, v, None, false, None);
        out.into_data().assert_approx_eq::<FT>(
            &TensorData::from([[[2.0, 3.0], [2.0, 3.0]]]),
            Tolerance::default(),
        );
    }

    #[test]
    fn sdpa_is_causal() {
        let (q, k, v) = qkv();
        let out = scaled_dot_product_attention(q, k, v, None, true, None);
        out.into_data().assert_approx_eq::<FT>(
            &TensorData::from([[[1.0, 2.0], [2.0, 3.0]]]),
            Tolerance::default(),
        );
    }

    #[test]
    fn sdpa_additive_mask() {
        let device = Default::default();
        let (q, k, v) = qkv();
        // Block query 0 from attending to key 1.
        let mask = Tensor::<3>::from_data(
            TensorData::from([[[0.0, f32::NEG_INFINITY], [0.0, 0.0]]]),
            &device,
        );
        let out = scaled_dot_product_attention(q, k, v, Some(mask), false, None);
        out.into_data().assert_approx_eq::<FT>(
            &TensorData::from([[[1.0, 2.0], [2.0, 3.0]]]),
            Tolerance::default(),
        );
    }
}
