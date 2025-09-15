use crate as burn;

use crate::module::{Content, DisplaySettings, Module, ModuleDisplay};
use crate::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use crate::{
    config::Config,
    tensor::{Tensor, backend::Backend},
};

use burn_tensor::activation::{quiet_softmax, softmax};

/// Window selection policy for streaming attention.
#[derive(Debug, Clone, Copy)]
pub enum AttnWindow {
    /// Attend to all cached tokens (full causal attention over cache).
    Full,
    /// Attend to at most `window_len` most recent tokens plus `sink_tokens`.
    Window(usize),
}

/// Configuration for the streaming multi-head attention module.
#[derive(Config, Debug)]
pub struct StreamingMultiHeadAttentionConfig {
    /// The size of the input/output features (d_model).
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dropout probability.
    #[config(default = 0.0)]
    pub dropout: f64,
    /// Use quiet softmax instead of regular softmax.
    #[config(default = false)]
    pub quiet_softmax: bool,
    /// Parameter initializer for linear layers.
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Streaming multi-head attention with KV cache and sliding window.
///
/// This module mirrors the projections of standard MultiHeadAttention but adds
/// a cache-managed forward path with a rolling KV buffer and an attention window
/// with optional sink tokens.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct StreamingMultiHeadAttention<B: Backend> {
    /// Query projection.
    pub query: Linear<B>,
    /// Key projection.
    pub key: Linear<B>,
    /// Value projection.
    pub value: Linear<B>,
    /// Output projection.
    pub output: Linear<B>,
    /// Dropout applied to attention scores.
    pub dropout: Dropout,
    /// Model dimension (per token).
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Head dimension (`d_model / n_heads`).
    pub d_k: usize,
    /// Use quiet softmax for attention weights.
    pub quiet_softmax: bool,
}

impl<B: Backend> ModuleDisplay for StreamingMultiHeadAttention<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("n_heads", &self.n_heads)
            .add("d_k", &self.d_k)
            .add("dropout", &self.dropout.prob)
            .add("quiet_softmax", &self.quiet_softmax)
            .optional()
    }
}

impl StreamingMultiHeadAttentionConfig {
    /// Initialize a new streaming multi-head attention module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> StreamingMultiHeadAttention<B> {
        let linear = |in_features, out_features| {
            LinearConfig::new(in_features, out_features)
                .with_initializer(self.initializer.clone())
                .init(device)
        };

        assert!(
            self.d_model % self.n_heads == 0,
            "d_model must be divisible by n_heads"
        );
        let d_k = self.d_model / self.n_heads;

        StreamingMultiHeadAttention {
            query: linear(self.d_model, self.d_model),
            key: linear(self.d_model, self.d_model),
            value: linear(self.d_model, self.d_model),
            output: linear(self.d_model, self.d_model),
            dropout: DropoutConfig::new(self.dropout).init(),
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_k,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

/// Streaming KV cache for sliding-window attention.
pub struct StreamingMhaCache<B: Backend> {
    /// Key buffer shaped `[batch, cache_len, n_heads, head_dim]`.
    pub k: Tensor<B, 4>,
    /// Value buffer shaped `[batch, cache_len, n_heads, head_dim]`.
    pub v: Tensor<B, 4>,
    /// Absolute token index after the latest write (exclusive).
    pub global_end_index: usize,
    /// Local (buffer) end index after the latest write (exclusive).
    pub local_end_index: usize,
    /// Number of sink tokens kept at the beginning of the buffer.
    pub sink_tokens: usize,
    /// Maximum length of the rolling buffer (capacity).
    pub cache_len: usize,
}

impl<B: Backend> StreamingMhaCache<B> {
    /// Create an empty cache with given capacity.
    pub fn new(
        device: &B::Device,
        batch: usize,
        cache_len: usize,
        n_heads: usize,
        head_dim: usize,
        sink_tokens: usize,
    ) -> Self {
        let zeros_k = Tensor::<B, 4>::zeros([batch, cache_len, n_heads, head_dim], device);
        let zeros_v = Tensor::<B, 4>::zeros([batch, cache_len, n_heads, head_dim], device);

        Self {
            k: zeros_k,
            v: zeros_v,
            global_end_index: 0,
            local_end_index: 0,
            sink_tokens,
            cache_len,
        }
    }

    /// Resets indices while keeping allocated buffers.
    pub fn reset(&mut self) {
        self.global_end_index = 0;
        self.local_end_index = 0;
    }

    /// Current number of valid tokens stored in the cache.
    pub fn len(&self) -> usize {
        self.local_end_index
    }

    /// Whether the cache currently holds no tokens.
    pub fn is_empty(&self) -> bool {
        self.local_end_index == 0
    }

    /// Cache capacity (in tokens).
    pub fn capacity(&self) -> usize {
        self.cache_len
    }

    /// Whether the cache is full.
    pub fn is_full(&self) -> bool {
        self.local_end_index >= self.cache_len
    }

    /// Create a shallow clone of this cache (tensor handles are cloned, buffers are not reallocated).
    pub fn clone_shallow(&self) -> Self {
        Self {
            k: self.k.clone(),
            v: self.v.clone(),
            global_end_index: self.global_end_index,
            local_end_index: self.local_end_index,
            sink_tokens: self.sink_tokens,
            cache_len: self.cache_len,
        }
    }

    /// Zero-out K and V buffers in place.
    pub fn clear(&mut self) {
        let device = self.k.device();
        let [b, cap, h, d] = self.k.dims();
        self.k = Tensor::<B, 4>::zeros([b, cap, h, d], &device);
        self.v = Tensor::<B, 4>::zeros([b, cap, h, d], &device);
        self.reset();
    }
}

/// Parameters for streaming attention forward.
pub struct StreamingParams<'a, B: Backend> {
    /// Optional rotary encoding to apply to Q and K, with an absolute start offset.
    pub rope: Option<&'a crate::nn::rope_encoding::RotaryEncoding<B>>,
    /// Absolute position of the first token in the current chunk.
    pub start_pos: usize,
    /// Window selection policy.
    pub window: AttnWindow,
}

impl<B: Backend> StreamingMultiHeadAttention<B> {
    /// Forward with streaming KV cache and optional windowing.
    ///
    /// Inputs are self-attention by default (query=key=value).
    pub fn forward_streaming(
        &self,
        x: Tensor<B, 3>,
        cache: &mut StreamingMhaCache<B>,
        params: StreamingParams<B>,
    ) -> Tensor<B, 3> {
        // Basic invariants on cache/window sizes.
        debug_assert!(
            cache.sink_tokens <= cache.cache_len,
            "sink tokens exceed cache capacity"
        );
        let [batch_size, seq_len, _] = x.dims();

        // Project to Q, K, V and reshape to [batch, n_heads, seq, d_k].
        let q = self.attention_linear(x.clone(), &self.query);
        let k = self.attention_linear(x.clone(), &self.key);
        let v = self.attention_linear(x, &self.value);

        // Optionally apply RoPE with offset to Q and K (last two dims are [n_heads, d_k]).
        let (q, k) = if let Some(rope) = params.rope {
            debug_assert!(self.d_k % 2 == 0, "RoPE requires even head_dim");
            // reshape to [batch, seq, n_heads, d_k] for rope apply along sequence
            let q_rs = q.swap_dims(1, 2); // [batch, seq, n_heads, d_k]
            let k_rs = k.swap_dims(1, 2);
            let q_ro = rope.apply(q_rs, params.start_pos);
            let k_ro = rope.apply(k_rs, params.start_pos);
            (q_ro.swap_dims(1, 2), k_ro.swap_dims(1, 2))
        } else {
            (q, k)
        };

        // Update rolling cache with new K/V tokens, possibly evicting.
        let num_new = seq_len;
        let current_end = params.start_pos + num_new;
        debug_assert!(
            current_end >= cache.global_end_index,
            "start_pos must be non-decreasing"
        );
        let sink = cache.sink_tokens;
        let cap = cache.cache_len;

        let delta = current_end.saturating_sub(cache.global_end_index);
        let need = cache.local_end_index + delta;

        if need > cap {
            let num_evicted = need - cap;
            let num_rolled = cache.local_end_index.saturating_sub(num_evicted + sink);
            if num_rolled > 0 {
                let src_start = sink + num_evicted;
                let src_end = sink + num_evicted + num_rolled;
                // roll K
                let rolled = cache.k.clone().slice([
                    0..batch_size,
                    src_start..src_end,
                    0..self.n_heads,
                    0..self.d_k,
                ]);
                cache.k.inplace(|t| {
                    t.slice_assign(
                        [
                            0..batch_size,
                            sink..sink + num_rolled,
                            0..self.n_heads,
                            0..self.d_k,
                        ],
                        rolled,
                    )
                });
                // roll V
                let rolled_v = cache.v.clone().slice([
                    0..batch_size,
                    src_start..src_end,
                    0..self.n_heads,
                    0..self.d_k,
                ]);
                cache.v.inplace(|t| {
                    t.slice_assign(
                        [
                            0..batch_size,
                            sink..sink + num_rolled,
                            0..self.n_heads,
                            0..self.d_k,
                        ],
                        rolled_v,
                    )
                });
            }
            // new local end index after eviction
            cache.local_end_index = cache.local_end_index + delta - num_evicted;
        } else {
            cache.local_end_index += delta;
        }

        // Write new K,V at the end
        let local_end = cache.local_end_index;
        let local_start = local_end - num_new;
        let k_rs = k.swap_dims(1, 2); // [batch, seq, n_heads, d_k]
        let v_rs = v.swap_dims(1, 2);

        cache.k.inplace(|t| {
            t.slice_assign(
                [
                    0..batch_size,
                    local_start..local_end,
                    0..self.n_heads,
                    0..self.d_k,
                ],
                k_rs,
            )
        });
        cache.v.inplace(|t| {
            t.slice_assign(
                [
                    0..batch_size,
                    local_start..local_end,
                    0..self.n_heads,
                    0..self.d_k,
                ],
                v_rs,
            )
        });
        cache.global_end_index = current_end;

        // Determine the active window in the cache for attention.
        let active_len = match params.window {
            AttnWindow::Full => local_end,
            AttnWindow::Window(w) => {
                debug_assert!(
                    cache.sink_tokens + w <= cache.cache_len,
                    "window+sink exceeds cache capacity"
                );
                sink + w.min(local_end.saturating_sub(sink))
            }
        };
        let start = local_end.saturating_sub(active_len);

        // Compute attention: [B, nH, Tq, d_k] x [B, nH, Tk, d_k]^T
        let q_use = q;
        let k_win = cache
            .k
            .clone()
            .slice([
                0..batch_size,
                start..local_end,
                0..self.n_heads,
                0..self.d_k,
            ])
            .swap_dims(1, 2); // [B, nH, Tk, d_k]
        let v_win = cache
            .v
            .clone()
            .slice([
                0..batch_size,
                start..local_end,
                0..self.n_heads,
                0..self.d_k,
            ])
            .swap_dims(1, 2);

        let mut attn_scores = q_use
            .matmul(k_win.transpose())
            .div_scalar((self.d_k as f32).sqrt());
        attn_scores = self.dropout.forward(attn_scores);

        let weights = if self.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        };

        let context = weights.matmul(v_win);
        // [B, nH, Tq, d_k] -> [B, Tq, nH, d_k] -> [B, Tq, d_model]
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model]);
        self.output.forward(context)
    }

    fn attention_linear(&self, x: Tensor<B, 3>, linear: &Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}

// Tests are provided as integration tests in crates/burn-core/tests/attention_streaming.rs
