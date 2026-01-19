//! Windowed (Local) Self-Attention Module for Burn
//!
//! Features:
//! - Causal and bidirectional attention modes
//! - Quantization-Safe Masking (min_float)
//! - Sparse-Ready (quiet_softmax)
//! - Rolling KV Cache for autoregressive inference

use burn_core as burn;

use super::mask::generate_sliding_window_mask;
use crate::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::{
    config::Config,
    module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay},
    tensor::{
        Bool, Tensor,
        activation::{quiet_softmax, softmax},
        backend::Backend,
    },
};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Configuration to create a [WindowedAttention](WindowedAttention) layer using the [init function](WindowedAttentionConfig::init).
#[derive(Config, Debug)]
pub struct WindowedAttentionConfig {
    /// The size of each linear layer.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// Number of key/value heads. Set to 1 for MQA, `n_heads` for MHA (default), or another divisor of `n_heads` for GQA.
    #[config(default = "None")]
    pub n_heads_kv: Option<usize>,
    /// The window size for local attention.
    pub window_size: usize,
    /// Whether to use causal (unidirectional) masking. Default: true
    #[config(default = true)]
    pub causal: bool,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
    /// The minimum value a float can take. Default: -1.0e4
    #[config(default = -1.0e4)]
    pub min_float: f64,
    /// Use "quiet softmax" instead of regular softmax. Default: false
    #[config(default = false)]
    pub quiet_softmax: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Windowed (local) self-attention module.
///
/// Restricts attention to a sliding window around each position.
///
/// # Params
///
/// - `query`: [`Linear`] layer with `d_model` input and output features.
/// - `key`: [`Linear`] layer with `d_model` input and output features.
/// - `value`: [`Linear`] layer with `d_model` input and output features.
/// - `output`: [`Linear`] layer with `d_model` input and output features.
///
/// Should be created with [WindowedAttentionConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct WindowedAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    dropout: Dropout,
    d_model: usize,
    n_heads: usize,
    n_heads_kv: usize,
    d_k: usize,
    window_size: usize,
    causal: bool,
    min_float: f64,
    quiet_softmax: bool,
}

impl<B: Backend> ModuleDisplay for WindowedAttention<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("n_heads", &self.n_heads)
            .add("n_heads_kv", &self.n_heads_kv)
            .add("d_k", &self.d_k)
            .add("window_size", &self.window_size)
            .add("causal", &self.causal)
            .add("dropout", &self.dropout.prob)
            .optional()
    }
}

impl WindowedAttentionConfig {
    /// Initialize a new [WindowedAttention](WindowedAttention) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> WindowedAttention<B> {
        assert_eq!(
            self.d_model % self.n_heads,
            0,
            "d_model must be divisible by n_heads"
        );

        let n_heads_kv = self.n_heads_kv.unwrap_or(self.n_heads);
        assert_eq!(
            self.n_heads % n_heads_kv,
            0,
            "n_heads must be divisible by n_heads_kv"
        );

        let d_k = self.d_model / self.n_heads;

        let query = LinearConfig::new(self.d_model, self.d_model)
            .with_initializer(self.initializer.clone())
            .init(device);
        let key = LinearConfig::new(self.d_model, n_heads_kv * d_k)
            .with_initializer(self.initializer.clone())
            .init(device);
        let value = LinearConfig::new(self.d_model, n_heads_kv * d_k)
            .with_initializer(self.initializer.clone())
            .init(device);
        let output = LinearConfig::new(self.d_model, self.d_model)
            .with_initializer(self.initializer.clone())
            .init(device);

        WindowedAttention {
            query,
            key,
            value,
            output,
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
            n_heads: self.n_heads,
            n_heads_kv,
            d_k,
            window_size: self.window_size,
            causal: self.causal,
            min_float: self.min_float,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

impl<B: Backend> WindowedAttention<B> {
    /// Applies windowed self-attention.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `[batch_size, seq_length, d_model]`.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch_size, seq_length, d_model]`.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_mask(input, None)
    }

    /// Applies windowed self-attention with an optional padding mask.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `[batch_size, seq_length, d_model]`.
    /// * `mask_pad` - Optional padding mask of shape `[batch_size, seq_length]` where `true` indicates positions to mask.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch_size, seq_length, d_model]`.
    pub fn forward_mask(
        &self,
        input: Tensor<B, 3>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length, d_model] = input.dims();
        let device = input.device();

        // 1. Project Q, K, V
        let query = self.reshape_query(self.query.forward(input.clone()));
        let key = self.reshape_kv(self.key.forward(input.clone()));
        let value = self.reshape_kv(self.value.forward(input));

        // 2. GQA expansion: repeat K, V heads to match Q heads
        let (key, value) = if self.n_heads != self.n_heads_kv {
            let n_rep = self.n_heads / self.n_heads_kv;
            (self.repeat_kv(key, n_rep), self.repeat_kv(value, n_rep))
        } else {
            (key, value)
        };

        // 3. Compute attention scores
        let attn_scores = query
            .matmul(key.transpose())
            .div_scalar((self.d_k as f32).sqrt());

        // 4. Apply sliding window mask
        let window_mask = generate_sliding_window_mask(
            batch_size,
            seq_length,
            self.window_size,
            self.causal,
            &device,
        );
        let attn_scores = attn_scores.mask_fill(
            window_mask.reshape([batch_size, 1, seq_length, seq_length]),
            self.min_float,
        );

        // 5. Apply optional padding mask
        let attn_scores = if let Some(mask_pad) = mask_pad {
            attn_scores.mask_fill(
                mask_pad.reshape([batch_size, 1, 1, seq_length]),
                self.min_float,
            )
        } else {
            attn_scores
        };

        // 6. Compute attention weights and apply dropout
        let weights = if self.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        };
        let weights = self.dropout.forward(weights);

        // 7. Compute context and project output
        let context = weights
            .matmul(value)
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length, d_model]);

        self.output.forward(context)
    }

    /// Applies windowed self-attention with a rolling KV cache for autoregressive inference.
    ///
    /// Only caches the last `window_size` KV pairs.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape `[batch_size, 1, d_model]` (single token).
    /// * `cache` - Mutable reference to the KV cache.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch_size, 1, d_model]`.
    pub fn forward_cache(
        &self,
        input: Tensor<B, 3>,
        cache: &mut WindowedAttentionCache<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length, d_model] = input.dims();
        debug_assert_eq!(seq_length, 1, "Cached inference expects single token input");

        // 1. Project Q, K, V for new token
        let query = self.reshape_query(self.query.forward(input.clone()));
        let new_key = self.reshape_kv(self.key.forward(input.clone()));
        let new_value = self.reshape_kv(self.value.forward(input));

        // 2. Update cache and get windowed KV
        let (key, value) = cache.update(new_key, new_value, self.window_size);

        // 3. GQA expansion: repeat K, V heads to match Q heads
        let (key, value) = if self.n_heads != self.n_heads_kv {
            let n_rep = self.n_heads / self.n_heads_kv;
            (self.repeat_kv(key, n_rep), self.repeat_kv(value, n_rep))
        } else {
            (key, value)
        };

        // 4. Compute attention scores
        let attn_scores = query
            .matmul(key.transpose())
            .div_scalar((self.d_k as f32).sqrt());

        // 5. Compute attention weights and apply dropout
        let weights = if self.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        };
        let weights = self.dropout.forward(weights);

        // 6. Compute context and project output
        let context = weights
            .matmul(value)
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length, d_model]);

        self.output.forward(context)
    }

    /// Reshape query projection to [batch, n_heads, seq_len, d_k].
    fn reshape_query(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _] = x.dims();
        x.reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }

    /// Reshape key/value projection to [batch, n_heads_kv, seq_len, d_k].
    fn reshape_kv(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _] = x.dims();
        x.reshape([batch_size, seq_length, self.n_heads_kv, self.d_k])
            .swap_dims(1, 2)
    }

    /// Repeat key/value heads for GQA: [b, h_kv, l, d] -> [b, h_kv * n_rep, l, d].
    fn repeat_kv(&self, x: Tensor<B, 4>, n_rep: usize) -> Tensor<B, 4> {
        let [b, h, l, d] = x.dims();
        x.reshape([b, h, 1, l, d])
            .expand([b, h, n_rep, l, d])
            .reshape([b, h * n_rep, l, d])
    }
}

/// Rolling KV cache for windowed attention autoregressive inference.
///
/// Only stores the last `window_size` key-value pairs, automatically evicting older entries as new tokens are processed.
pub struct WindowedAttentionCache<B: Backend> {
    key: Option<Tensor<B, 4>>,
    value: Option<Tensor<B, 4>>,
}

impl<B: Backend> WindowedAttentionCache<B> {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            key: None,
            value: None,
        }
    }

    fn update(
        &mut self,
        new_key: Tensor<B, 4>,
        new_value: Tensor<B, 4>,
        window_size: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let truncate = |t: Tensor<B, 4>, window: usize| -> Tensor<B, 4> {
            let seq_len = t.dims()[2];
            if seq_len > window {
                t.narrow(2, seq_len - window, window)
            } else {
                t
            }
        };

        let key = match self.key.take() {
            Some(cached) => truncate(Tensor::cat(vec![cached, new_key], 2), window_size),
            None => new_key,
        };

        let value = match self.value.take() {
            Some(cached) => truncate(Tensor::cat(vec![cached, new_value], 2), window_size),
            None => new_value,
        };

        self.key = Some(key.clone());
        self.value = Some(value.clone());

        (key, value)
    }
}

impl<B: Backend> Default for WindowedAttentionCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{Distribution, Shape, Tolerance};

    #[test]
    fn test_windowed_attention_shapes() {
        let [batch_size, seq_length, d_model, n_heads, window_size] = [2, 8, 32, 4, 2];
        let device = Default::default();

        let config = WindowedAttentionConfig::new(d_model, n_heads, window_size);
        let module = config.init::<TestBackend>(&device);

        let input = Tensor::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let output = module.forward(input);

        assert_eq!(
            output.shape(),
            Shape::new([batch_size, seq_length, d_model])
        );
    }

    #[test]
    fn test_windowed_attention_bidirectional() {
        let [batch_size, seq_length, d_model, n_heads, window_size] = [2, 8, 32, 4, 2];
        let device = Default::default();

        let config = WindowedAttentionConfig::new(d_model, n_heads, window_size).with_causal(false);
        let module = config.init::<TestBackend>(&device);

        let input = Tensor::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let output = module.forward(input);

        assert_eq!(
            output.shape(),
            Shape::new([batch_size, seq_length, d_model])
        );
    }

    #[test]
    fn test_windowed_attention_masking_correctness() {
        let [batch_size, seq_length, d_model, n_heads, window_size] = [1, 6, 16, 2, 1];
        let device = Default::default();

        let config = WindowedAttentionConfig::new(d_model, n_heads, window_size).with_dropout(0.0);
        let module = config.init::<TestBackend>(&device);

        let input = Tensor::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let output1 = module.forward(input.clone());

        // Modify position 0, which is outside the window of position 3
        // With window_size=1 and causal=true, position 3 only attends to positions 2,3
        let mut modified = input.clone();
        modified = modified.slice_assign(
            [0..1, 0..1, 0..d_model],
            Tensor::random([1, 1, d_model], Distribution::Default, &device),
        );
        let output2 = module.forward(modified);

        // Position 3's output should be unchanged
        let pos3_out1 = output1.clone().slice([0..1, 3..4, 0..d_model]);
        let pos3_out2 = output2.clone().slice([0..1, 3..4, 0..d_model]);

        pos3_out1
            .into_data()
            .assert_approx_eq(&pos3_out2.into_data(), Tolerance::<f32>::default());
    }

    #[test]
    fn test_windowed_attention_padding_mask() {
        let [batch_size, seq_length, d_model, n_heads, window_size] = [1, 4, 16, 2, 2];
        let device = Default::default();

        let config = WindowedAttentionConfig::new(d_model, n_heads, window_size)
            .with_dropout(0.0)
            .with_causal(false);
        let module = config.init::<TestBackend>(&device);

        let input = Tensor::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let mask = Tensor::from_data([[false, false, true, true]], &device);

        let output1 = module.forward_mask(input.clone(), Some(mask.clone()));

        // Modify masked positions
        let mut modified = input.clone();
        modified = modified.slice_assign(
            [0..1, 2..4, 0..d_model],
            Tensor::random([1, 2, d_model], Distribution::Default, &device),
        );
        let output2 = module.forward_mask(modified, Some(mask));

        // Unmasked positions should produce same output
        let unmasked1 = output1.slice([0..1, 0..2, 0..d_model]);
        let unmasked2 = output2.slice([0..1, 0..2, 0..d_model]);

        unmasked1
            .into_data()
            .assert_approx_eq(&unmasked2.into_data(), Tolerance::<f32>::default());
    }

    #[test]
    fn test_windowed_attention_cache() {
        let [batch_size, d_model, n_heads, window_size] = [2, 32, 4, 3];
        let device = Default::default();

        let config = WindowedAttentionConfig::new(d_model, n_heads, window_size);
        let module = config.init::<TestBackend>(&device);

        let mut cache = WindowedAttentionCache::new();

        for _ in 0..5 {
            let input = Tensor::random([batch_size, 1, d_model], Distribution::Default, &device);
            let output = module.forward_cache(input, &mut cache);
            assert_eq!(output.shape(), Shape::new([batch_size, 1, d_model]));
        }

        let cached_len = cache.key.as_ref().unwrap().dims()[2];
        assert_eq!(cached_len, window_size);
    }

    #[test]
    fn test_windowed_attention_cache_equivalence() {
        let [batch_size, d_model, n_heads, window_size] = [1, 16, 2, 4];
        let device = Default::default();

        let config = WindowedAttentionConfig::new(d_model, n_heads, window_size).with_dropout(0.0);
        let module = config.init::<TestBackend>(&device);

        let seq_len = 3;
        let full_input = Tensor::random(
            [batch_size, seq_len, d_model],
            Distribution::Default,
            &device,
        );

        // Forward with full sequence
        let full_output = module.forward(full_input.clone());

        // Forward token by token with cache
        let mut cache = WindowedAttentionCache::new();
        let mut cached_outputs = Vec::new();
        for i in 0..seq_len {
            let token = full_input.clone().slice([0..1, i..i + 1, 0..d_model]);
            let out = module.forward_cache(token, &mut cache);
            cached_outputs.push(out);
        }

        // Compare last token output
        let full_last = full_output.slice([0..1, (seq_len - 1)..seq_len, 0..d_model]);
        let cached_last = cached_outputs.last().unwrap().clone();

        full_last
            .into_data()
            .assert_approx_eq(&cached_last.into_data(), Tolerance::<f32>::default());
    }

    #[test]
    #[should_panic(expected = "d_model must be divisible by n_heads")]
    fn test_windowed_attention_invalid_config() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let config = WindowedAttentionConfig::new(32, 5, 2);
        config.init::<TestBackend>(&device);
    }

    #[test]
    fn test_windowed_attention_gqa_shapes() {
        let [batch_size, seq_length, d_model, n_heads, n_heads_kv, window_size] = [2, 8, 32, 4, 2, 2];
        let device = Default::default();

        let config = WindowedAttentionConfig::new(d_model, n_heads, window_size)
            .with_n_heads_kv(Some(n_heads_kv));
        let module = config.init::<TestBackend>(&device);

        let input = Tensor::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let output = module.forward(input);

        assert_eq!(
            output.shape(),
            Shape::new([batch_size, seq_length, d_model])
        );
    }

    #[test]
    fn test_windowed_attention_mqa_shapes() {
        let [batch_size, seq_length, d_model, n_heads, window_size] = [2, 8, 32, 4, 2];
        let device = Default::default();

        let config = WindowedAttentionConfig::new(d_model, n_heads, window_size)
            .with_n_heads_kv(Some(1)); // MQA
        let module = config.init::<TestBackend>(&device);

        let input = Tensor::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let output = module.forward(input);

        assert_eq!(
            output.shape(),
            Shape::new([batch_size, seq_length, d_model])
        );
    }

    #[test]
    #[should_panic(expected = "n_heads must be divisible by n_heads_kv")]
    fn test_gqa_panic_if_n_heads_not_divisible_by_n_heads_kv() {
        let device: <TestBackend as Backend>::Device = Default::default();
        // d_model=32, n_heads=4 is valid (32/4=8), but n_heads=4, n_heads_kv=3 is invalid (4%3!=0)
        let config = WindowedAttentionConfig::new(32, 4, 2).with_n_heads_kv(Some(3));
        config.init::<TestBackend>(&device);
    }

    #[test]
    fn display() {
        let [d_model, n_heads, window_size] = [32, 4, 2];
        let device = Default::default();

        let config = WindowedAttentionConfig::new(d_model, n_heads, window_size);
        let module = config.init::<TestBackend>(&device);

        let s = alloc::format!("{}", module);
        assert!(s.contains("d_model: 32"));
        assert!(s.contains("n_heads: 4"));
        assert!(s.contains("n_heads_kv: 4"));
        assert!(s.contains("window_size: 2"));
        assert!(s.contains("causal: true"));
    }
}
