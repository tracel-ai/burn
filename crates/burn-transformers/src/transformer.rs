use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, RotaryEncoding,
        SwiGlu, SwiGluConfig,
    },
    tensor::{activation::softmax, backend::Backend, Bool, Device, Int, Tensor},
};

use crate::cache::AutoregressiveCache;

/// Configuration to create a Llama [decoder-only transformer](Transformer).
#[derive(Config)]
pub struct TransformerConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// Maximum token sequence length.
    #[config(default = "512")]
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    #[config(default = "1e-5")]
    pub norm_eps: f64,
}

impl TransformerConfig {
    /// Initialize a new [decoder-only transformer](Transformer).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Transformer<B> {
        let tok_embeddings = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let layers = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(
                    self.n_layers,
                    self.d_model,
                    self.hidden_size,
                    self.n_heads,
                    self.n_kv_heads,
                    self.norm_eps,
                )
                .init(device)
            })
            .collect::<Vec<_>>();
        let norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let output = LinearConfig::new(self.d_model, self.vocab_size)
            .with_bias(false)
            .init(device);

        Transformer {
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }
}

/// Llama decoder-only transformer.
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    tok_embeddings: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    // NOTE: Starting with Llama 3.2, the weights of the output layer are tied with the embedding
    output: Linear<B>,
}

impl<B: Backend> Transformer<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        cache: &mut Vec<KeyValueCache<B>>,
        rope: &RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let mut h = self.tok_embeddings.forward(input);

        for (layer, c) in self.layers.iter().zip(cache.into_iter()) {
            h = layer.forward(h, c, rope);
        }

        let h = self.norm.forward(h);
        self.output.forward(h)
    }
}

/// Configuration to create a [decoder-only transformer block](TransformerBlock).
#[derive(Config)]
pub struct TransformerBlockConfig {
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl TransformerBlockConfig {
    /// Initialize a new [decoder-only transformer block](TransformerBlock).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> TransformerBlock<B> {
        let attention =
            MultiHeadAttentionConfig::new(self.d_model, self.n_heads, self.n_kv_heads).init(device);
        let feed_forward = FeedForwardConfig::new(self.d_model, self.hidden_size).init(device);
        let attention_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let ffn_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);

        TransformerBlock {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        }
    }
}

/// Decoder-only transformer block.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Self-attention.
    attention: MultiHeadAttention<B>,
    /// Feed-forward transformation.
    feed_forward: FeedForward<B>,
    /// Attention pre-normalization.
    attention_norm: RmsNorm<B>,
    /// Feed-forward pre-normalization.
    ffn_norm: RmsNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        cache: &mut KeyValueCache<B>,
        rope: &RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let h = input.clone()
            + self
                .attention
                .forward(self.attention_norm.forward(input), cache, rope);
        h.clone() + self.feed_forward.forward(self.ffn_norm.forward(h))
    }
}

/// Configuration to create a [feed-forward transformation network](FeedForward).
#[derive(Config)]
pub struct FeedForwardConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub hidden_size: usize,
}

impl FeedForwardConfig {
    /// Initialize a new [feed-forward transformation network](FeedForward).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> FeedForward<B> {
        let swiglu = SwiGluConfig::new(self.d_model, self.hidden_size)
            .with_bias(false)
            .init(device);
        let w2 = LinearConfig::new(self.hidden_size, self.d_model)
            .with_bias(false)
            .init(device);

        FeedForward { swiglu, w2 }
    }
}

/// Feed-forward transformation network.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    // Swish gated linear unit with trainable parameters.
    swiglu: SwiGlu<B>,
    /// Outer linear.
    w2: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.w2.forward(self.swiglu.forward(input))
    }
}

/// Key-value cache for autoregressive models.
pub struct KeyValueCache<B: Backend> {
    key: AutoregressiveCache<B>,
    value: AutoregressiveCache<B>,
}

impl<B: Backend> KeyValueCache<B> {
    /// Create a new [key-value cache](KeyValueCache).
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            key: AutoregressiveCache::new(max_batch_size, num_heads, max_seq_len, d_model, device),
            value: AutoregressiveCache::new(
                max_batch_size,
                num_heads,
                max_seq_len,
                d_model,
                device,
            ),
        }
    }

    /// Computes the complete keys and values.
    pub fn forward(
        &mut self,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k = self.key.forward(key);
        let v = self.value.forward(value);
        (k, v)
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        // We can assume key and value have the same length
        self.key.len()
    }

    /// Reset key-value cache.
    /// Use between different contexts (i.e., for each new prompt).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }
}

/// Configuration to create a [multi-head attention](MultiHeadAttention) module.
#[derive(Config)]
pub struct MultiHeadAttentionConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
}

impl MultiHeadAttentionConfig {
    /// Initialize a new [multi-head attention](MultiHeadAttention) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> MultiHeadAttention<B> {
        let head_dim = self.d_model / self.n_heads;

        let wq = LinearConfig::new(self.d_model, self.n_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wk = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wv = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wo = LinearConfig::new(self.n_heads * head_dim, self.d_model)
            .with_bias(false)
            .init(device);

        MultiHeadAttention {
            wq,
            wk,
            wv,
            wo,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Query projection.
    wq: Linear<B>,
    /// Key projection.
    wk: Linear<B>,
    /// Value projection.
    wv: Linear<B>,
    /// Output projection.
    wo: Linear<B>,

    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Applies the forward pass on the input tensors.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        cache: &mut KeyValueCache<B>,
        rope: &RotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        let device = input.device();
        let [batch_size, seq_len, hidden_size] = input.dims();

        let q = self.wq.forward(input.clone());
        let k = self.wk.forward(input.clone());
        let v = self.wv.forward(input);

        // [batch_size, num_heads, seq_len, head_dim]
        let q = q
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        // Sequence start position can be deduced from the number of cached items
        let cache_seq_len = cache.len();

        let q = rope.apply(q, cache_seq_len);
        let k = rope.apply(k, cache_seq_len);

        // Key-value caching
        let (k, v) = cache.forward(k, v);

        // Repeat key/value heads if num_kv_heads < num_heads
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        // Attention scores
        let mut scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((self.head_dim as f32).sqrt());

        // Matrix of scores is of size [seqlen, cache_len + seqlen], and the only masked entries are
        // (i, j) for j > cache_len + i, since row i corresponds to token cache_len + i.
        // NOTE: we could possibly improve the mask generation by caching masks for different sequence lengths,
        // though it is probably not necessary at this time.
        if seq_len > 1 {
            let cache_seq_len = cache.len();
            let mask = Tensor::<B, 2, Bool>::tril_mask(
                [seq_len, cache_seq_len],
                (cache_seq_len - seq_len) as i64, // offset
                &device,
            );
            scores = scores.mask_fill(mask.unsqueeze::<4>(), f32::NEG_INFINITY);
        }

        let scores = softmax(scores, 3);

        // Output [batch_size, num_heads, seq_len, head_dim]
        let output = scores.matmul(v);
        let output = output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, hidden_size]);
        self.wo.forward(output)
    }

    /// Repeats a key or value tensor for grouped query attention.
    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            x
        } else {
            let [batch_size, num_kv_heads, seq_len, head_dim] = x.dims();

            x.unsqueeze_dim::<5>(2)
                .expand([batch_size, num_kv_heads, n_rep, seq_len, head_dim])
                .reshape([batch_size, num_kv_heads * n_rep, seq_len, head_dim])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    use burn::tensor::TensorData;

    #[test]
    fn test_rms_norm() {
        let device = Default::default();

        let rms = RmsNormConfig::new(4).with_epsilon(1e-5).init(&device);
        let input = TestTensor::<3>::from([[
            [0.0025997162, 0.0030002594, -0.006000519, 0.006000519],
            [0.0010004044, 0.00080013275, 0.0015001297, -0.01600647],
        ]]);

        let output = rms.forward(input);
        let expected = TensorData::from([[
            [0.45996094, 0.5307617, -1.0615234, 1.0615234],
            [0.11553955, 0.09240723, 0.17321777, -1.8486328],
        ]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}
