use crate::Initializer;
use burn_core as burn;

use crate::activation::Gelu;
use crate::cache::{CacheState, TensorCache};
use crate::modules::bounds;
use crate::modules::role::Role;
use crate::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::config::Config;
use burn::instrument::{Bounds, compute};
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::{Bool, Device, Tensor, assert_shape};

use burn::tensor::activation::{quiet_softmax, softmax};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Configuration to create a [Multi Head Attention](MultiHeadAttention) layer using the [init function](MultiHeadAttentionConfig::init).
#[derive(Config, Debug)]
pub struct MultiHeadAttentionConfig {
    /// The size of each linear layer.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Legacy masking value, retained for configuration compatibility. Default: -1.0e4
    /// Attention masking uses negative infinity; this value has no effect.
    #[config(default = -1.0e4)]
    pub min_float: f64,
    /// Use "quiet softmax" instead of regular softmax.
    ///
    /// - Usage may improve performance by allowing attention heads to deposit no information (if the sequence contains no information relevant to that head).
    /// - Usage may reduce the entropy of weights in the model, enhancing quantization and compression.
    ///
    /// Reference: <https://www.evanmiller.org/attention-is-off-by-one.html>
    #[config(default = false)]
    pub quiet_softmax: bool,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,

    /// Enable bias for the key [Linear] layer.
    #[config(default = true)]
    pub key_bias: bool,

    /// Enable bias for the value [Linear] layer.
    #[config(default = true)]
    pub value_bias: bool,

    /// Enable bias for the query [Linear] layer.
    #[config(default = true)]
    pub query_bias: bool,

    /// Enable bias for the output [Linear] layer.
    #[config(default = true)]
    pub output_bias: bool,
}

/// The multihead attention module as describe in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
///
/// # Params
///
/// - `query`: [`Linear`] layer with `d_model` input and output features.
/// - `key`: [`Linear`] layer with `d_model` input and output features.
/// - `value`: [`Linear`] layer with `d_model` input and output features.
/// - `output`: [`Linear`] layer with `d_model` input and output features.
///
/// Should be created with [MultiHeadAttentionConfig].
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct MultiHeadAttention {
    /// Linear layer to transform the input features into the query space.
    pub query: Linear,
    /// Linear layer to transform the input features into the key space.
    pub key: Linear,
    /// Linear layer to transform the input features into the value space.
    pub value: Linear,
    /// Linear layer to transform the output features back to the original space.
    pub output: Linear,
    /// Dropout layer.
    pub dropout: Dropout,
    /// Activation function.
    pub activation: Gelu,
    /// The size of each linear layer.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// Size of the key and query vectors.
    pub d_k: usize,
    /// Legacy masking value, retained for compatibility. This value has no effect.
    pub min_float: f64,
    /// Use "quiet softmax" instead of regular softmax.
    pub quiet_softmax: bool,
}

impl ModuleDisplay for MultiHeadAttention {
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
            .add("min_float", &self.min_float)
            .add("quiet_softmax", &self.quiet_softmax)
            .optional()
    }
}

/// [Multihead attention](MultiHeadAttention) forward pass input argument.
#[derive(Debug, Clone)]
pub struct MhaInput {
    /// Shape `[batch_size, seq_length_1, d_model]`
    query: Tensor<3>,
    /// Shape `[batch_size, seq_length_2, d_model]`
    key: Tensor<3>,
    /// Shape `[batch_size, seq_length_2, d_model]`
    value: Tensor<3>,
    mask_pad: Option<Tensor<2, Bool>>,
    mask_attn: Option<Tensor<3, Bool>>,
}

impl MultiHeadAttentionConfig {
    /// Initialize a new [multihead attention](MultiHeadAttention) module.
    pub fn init(&self, device: &Device) -> MultiHeadAttention {
        let linear_cfg = LinearConfig::new(self.d_model, self.d_model)
            .with_initializer(self.initializer.clone());

        MultiHeadAttention {
            query: linear_cfg.clone().with_bias(self.query_bias).init(device),
            key: linear_cfg.clone().with_bias(self.key_bias).init(device),
            value: linear_cfg.clone().with_bias(self.value_bias).init(device),
            output: linear_cfg.clone().with_bias(self.output_bias).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Gelu::new(),
            n_heads: self.n_heads,
            d_k: self.d_model / self.n_heads,
            min_float: self.min_float,
            quiet_softmax: self.quiet_softmax,
            d_model: self.d_model,
        }
    }
}

impl MhaInput {
    /// Create a [multihead attention](MultiHeadAttention) input argument
    /// by setting the query, key and value to the given tensor.
    ///
    /// # Shape
    /// - tensor: `[batch_size, seq_length, d_model]`
    pub fn self_attn(tensor: Tensor<3>) -> Self {
        Self {
            query: tensor.clone(),
            key: tensor.clone(),
            value: tensor,
            mask_pad: None,
            mask_attn: None,
        }
    }

    /// Create a [multihead attention](MultiHeadAttention) input argument.
    pub fn new(query: Tensor<3>, key: Tensor<3>, value: Tensor<3>) -> Self {
        Self {
            query,
            key,
            value,
            mask_pad: None,
            mask_attn: None,
        }
    }

    /// Register the padding mask.
    pub fn mask_pad(mut self, mask_pad: Tensor<2, Bool>) -> Self {
        self.mask_pad = Some(mask_pad);
        self
    }

    /// Register the attention mask.
    pub fn mask_attn(mut self, mask_attn: Tensor<3, Bool>) -> Self {
        self.mask_attn = Some(mask_attn);
        self
    }
}

/// [Multihead attention](MultiHeadAttention) outputs.
#[derive(Debug, Clone)]
pub struct MhaOutput {
    /// The attention weights `[batch_size, n_heads, seq_length_1, seq_length_2]`.
    /// Masked entries, including fully masked rows, have zero weight.
    /// During training, dropout is applied to these weights after softmax, so rows may not sum to 1.
    pub weights: Tensor<4>,
    /// The context tensor `[batch_size, seq_length_1, d_model]`.
    pub context: Tensor<3>,
}

impl MultiHeadAttention {
    /// Applies the forward pass on the input tensors.
    ///
    /// See [MultiHeadAttention](MultiHeadAttention) for more information.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    ///
    /// # Panics
    ///
    /// Panics if the last axis of `query`, `key` or `value` is not `d_model`, if the batch size
    /// of `key` or `value` differs from `query`, or if the sequence length of `value` differs
    /// from `key`.
    #[compute]
    pub fn forward(&self, input: MhaInput) -> MhaOutput {
        let [batch_size, seq_length_1, d_model] = input.query.dims();
        let [_, seq_length_2, _] = input.key.dims();
        assert_shape!(input.query, [_, _, self.d_model]);
        assert_shape!(input.key, [batch_size, _, self.d_model]);
        assert_shape!(input.value, [batch_size, seq_length_2, self.d_model]);

        let query = self.attention_linear(input.query, Role::QUERY, &self.query);
        let key = self.attention_linear(input.key, Role::KEY, &self.key);
        let value = self.attention_linear(input.value, Role::VALUE, &self.value);

        let (weights, context) = self.attend(query, key, value, input.mask_pad, input.mask_attn);
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length_1, d_model]);
        let context = Role::OUTPUT.run(|| self.output.forward(context));

        MhaOutput { weights, context }
    }

    /// Applies the forward pass using a cache.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    ///
    /// # Panics
    ///
    /// Panics if the last axis of `query`, `key` or `value` is not `d_model`, if the batch size
    /// of `key` or `value` differs from `query`, or if the sequence length of `value` differs
    /// from `key`.
    pub fn forward_cache(&self, input: MhaInput, cache: &mut MhaCache) -> MhaOutput {
        self.forward_cache_inner(input, cache, false)
    }

    /// Applies cached attention, returning only the context tensor.
    ///
    /// Pass the full query prefix on each call: a nonempty prefill followed by exactly one
    /// appended token per call. Keys and values follow the same rule for an
    /// [autoregressive cache](MhaCache::autoregressive), or remain fixed for a
    /// [cross-attention cache](MhaCache::autoregressive_cross_attention).
    ///
    /// After prefill, only the newest query attends to the cached keys and values. The returned
    /// tensor includes the cached output prefix, just like [`forward_cache`](Self::forward_cache).
    /// Attention masks use the full prefix shape; only their newest query row is evaluated.
    /// Previously cached outputs are retained even if earlier input tokens or mask rows change.
    ///
    /// The same cache may be used interchangeably with [`forward_cache`](Self::forward_cache):
    /// query projections are retained so that full attention weights remain available. When
    /// dropout is active, this uses the full attention path to preserve its random draws.
    /// Floating-point results may differ slightly because the attention matrix shapes differ.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key, value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    ///
    /// # Panics
    ///
    /// Panics under the same shape constraints as [`forward_cache`](Self::forward_cache).
    pub fn forward_cache_context(&self, input: MhaInput, cache: &mut MhaCache) -> Tensor<3> {
        let dropout_active = self.dropout.prob > 0.0
            && self.dropout.training.is_enabled()
            && input.query.device().is_autodiff();
        let last_query_only = cache.output.is_initialized() && !dropout_active;

        self.forward_cache_inner(input, cache, last_query_only)
            .context
    }

    #[compute]
    fn forward_cache_inner(
        &self,
        mut input: MhaInput,
        cache: &mut MhaCache,
        last_query_only: bool,
    ) -> MhaOutput {
        let [batch_size, seq_length_1, d_model] = input.query.dims();
        let [_, seq_length_2, _] = input.key.dims();
        assert_shape!(input.query, [_, _, self.d_model]);
        assert_shape!(input.key, [batch_size, _, self.d_model]);
        assert_shape!(input.value, [batch_size, seq_length_2, self.d_model]);

        let mut query = cache.query.forward(input.query, |t| {
            self.attention_linear(t, Role::QUERY, &self.query)
        });
        let key = cache.key.forward(input.key, |t| {
            self.attention_linear(t, Role::KEY, &self.key)
        });
        let value = cache.value.forward(input.value, |t| {
            self.attention_linear(t, Role::VALUE, &self.value)
        });

        if last_query_only {
            assert_shape!(query, [batch_size, self.n_heads, seq_length_1, self.d_k]);
            query = query.slice([
                0..batch_size,
                0..self.n_heads,
                seq_length_1 - 1..seq_length_1,
                0..self.d_k,
            ]);
            input.mask_attn = input.mask_attn.map(|mask| {
                let [batch_size, query_length, key_length] = mask.dims();
                if query_length != 1 {
                    assert_shape!(mask, [_, seq_length_1, _]);
                }
                mask.slice([0..batch_size, query_length - 1..query_length, 0..key_length])
            });
        }

        let seq_length_1 = if last_query_only { 1 } else { seq_length_1 };
        let (weights, context) = self.attend(query, key, value, input.mask_pad, input.mask_attn);
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length_1, d_model]);

        let context = cache
            .output
            .forward(context, |t| Role::OUTPUT.run(|| self.output.forward(t)));

        MhaOutput { weights, context }
    }

    /// The arithmetic of attending from `query` over `key` and a value shaped like it, both
    /// `[batch_size, seq_length, d_model]`: the four projections and the scaled dot product
    /// between them. What a composite holding this attention declares of it.
    pub(crate) fn flops(&self, query: &Tensor<3>, key: &Tensor<3>) -> u64 {
        let [batch_size, seq_length_1, _] = query.dims();
        let [_, seq_length_2, _] = key.dims();
        // The output projection's input is the context, shaped like the query.
        let projections = self.query.bounds(query).flops
            + self.key.bounds(key).flops
            + self.value.bounds(key).flops
            + self.output.bounds(query).flops;
        let attention = bounds::ScaledDotProduct {
            query_rows: (batch_size * self.n_heads * seq_length_1) as u64,
            key_length: seq_length_2 as u64,
            key_width: self.d_k as u64,
            value_width: self.d_k as u64,
        };
        projections + attention.flops()
    }

    /// Weigh `value` by how well `query` matches `key`, returning the attention weights and the
    /// context, `[batch_size, n_heads, seq_length_1, d_k]`.
    #[compute(label = "scaled dot product", bounds = Self::attend_bounds)]
    fn attend(
        &self,
        query: Tensor<4>,
        key: Tensor<4>,
        value: Tensor<4>,
        mask_pad: Option<Tensor<2, Bool>>,
        mask_attn: Option<Tensor<3, Bool>>,
    ) -> (Tensor<4>, Tensor<4>) {
        let attn_scores = self.attn_scores(query, key);
        let weights = self.attn_weights(attn_scores, mask_pad, mask_attn);
        let context = weights.clone().matmul(value);
        (weights, context)
    }

    /// The floor for [`attend`](Self::attend) alone: scaled dot-product attention's, reading the
    /// masks besides and writing the weights, which are returned.
    fn attend_bounds(
        &self,
        query: &Tensor<4>,
        key: &Tensor<4>,
        value: &Tensor<4>,
        mask_pad: &Option<Tensor<2, Bool>>,
        mask_attn: &Option<Tensor<3, Bool>>,
    ) -> Bounds {
        let attention = bounds::attention(query, key, value);
        let masks = bounds::masks(mask_pad, mask_attn);
        let [batch_size, n_heads, seq_length_1, _] = query.dims().map(|dim| dim as u64);
        let seq_length_2 = key.dims()[2] as u64;
        let weights = batch_size * n_heads * seq_length_1 * seq_length_2;
        Bounds {
            read: attention.read + masks,
            write: attention.write + weights * query.dtype().size() as u64,
            ..attention
        }
    }

    fn attn_scores(&self, query: Tensor<4>, key: Tensor<4>) -> Tensor<4> {
        query
            .matmul(key.transpose())
            .div_scalar((self.d_k as f32).sqrt())
    }

    fn attn_weights(
        &self,
        mut attn_scores: Tensor<4>,
        mask_pad: Option<Tensor<2, Bool>>,
        mask_attn: Option<Tensor<3, Bool>>,
    ) -> Tensor<4> {
        let mask_pad = mask_pad.map(|mask_pad| {
            let [batch_size, seq_length] = mask_pad.dims();
            mask_pad.reshape([batch_size, 1, 1, seq_length])
        });
        let mask_attn = mask_attn.map(|mask_attn| {
            let [batch_size, seq_length_1, seq_length_2] = mask_attn.dims();
            mask_attn.reshape([batch_size, 1, seq_length_1, seq_length_2])
        });
        let mask = match (mask_pad, mask_attn) {
            (Some(mask_pad), Some(mask_attn)) => Some(mask_pad.bool_or(mask_attn)),
            (Some(mask), None) | (None, Some(mask)) => Some(mask),
            (None, None) => None,
        };

        if let Some(mask) = &mask {
            attn_scores = attn_scores.mask_fill(mask.clone(), f32::NEG_INFINITY);
            // Avoid NaNs in both forward and backward by making fully masked rows
            // finite before softmax. Their weights are explicitly zeroed below.
            attn_scores = attn_scores.mask_fill(mask.clone().all_dim(3), 0.0);
        }

        let weights = if self.quiet_softmax {
            quiet_softmax(attn_scores, 3)
        } else {
            softmax(attn_scores, 3)
        };

        let weights = if let Some(mask) = mask {
            weights.mask_fill(mask, 0.0)
        } else {
            weights
        };

        self.dropout.forward(weights)
    }

    fn attention_linear(&self, x: Tensor<3>, role: Role, linear: &Linear) -> Tensor<4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        role.run(|| linear.forward(x))
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}

/// Cache for the [Multi Head Attention](MultiHeadAttention) layer.
///
/// To be used during inference when decoding tokens.
pub struct MhaCache {
    query: MhaLinearCache<4>,
    key: MhaLinearCache<4>,
    value: MhaLinearCache<4>,
    output: MhaLinearCache<3>,
}

enum MhaLinearCache<const D: usize> {
    Autoregressive(TensorCache<D>, usize),
    Full(TensorCache<D>),
}

impl MhaCache {
    /// Initialize a cache for autoregressive inference.
    pub fn autoregressive() -> Self {
        Self {
            query: MhaLinearCache::Autoregressive(TensorCache::empty(), 2),
            key: MhaLinearCache::Autoregressive(TensorCache::empty(), 2),
            value: MhaLinearCache::Autoregressive(TensorCache::empty(), 2),
            output: MhaLinearCache::Autoregressive(TensorCache::empty(), 1),
        }
    }

    /// Initialize a cache for autoregressive inference, but with a fixed memory used for keys and
    /// values (cross-attention).
    pub fn autoregressive_cross_attention() -> Self {
        Self {
            query: MhaLinearCache::Autoregressive(TensorCache::empty(), 2),
            key: MhaLinearCache::Full(TensorCache::empty()),
            value: MhaLinearCache::Full(TensorCache::empty()),
            output: MhaLinearCache::Autoregressive(TensorCache::empty(), 1),
        }
    }
}

impl<const D: usize> MhaLinearCache<D> {
    fn is_initialized(&self) -> bool {
        let cache = match self {
            Self::Autoregressive(cache, _) | Self::Full(cache) => cache,
        };
        matches!(&cache.state, CacheState::Value(_))
    }

    pub fn forward<F: Fn(Tensor<3>) -> Tensor<D>>(
        &mut self,
        tensor: Tensor<3>,
        func: F,
    ) -> Tensor<D> {
        match self {
            MhaLinearCache::Autoregressive(cache, dim) => {
                cache.forward_autoregressive(tensor, *dim, func)
            }
            MhaLinearCache::Full(cache) => cache.forward_full(tensor, func),
        }
    }
}

#[cfg(test)]
#[path = "mha_cache_tests.rs"]
mod cache_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::generate_autoregressive_mask;
    use alloc::vec::Vec;
    use burn::tensor::Int;
    use burn::tensor::Tolerance;
    use burn::tensor::{DType, TensorData};
    use burn::tensor::{Distribution, Shape};
    use rstest::rstest;

    /// Each projection runs under its role, so the four do not merge into one, and the
    /// attention between them is priced on its own: reading the heads, writing the context and
    /// the weights it returns.
    #[cfg(feature = "std")]
    #[test]
    fn attention_names_its_projections_and_prices_its_core() {
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(8, 2).init(&device);
        let input = MhaInput::self_attn(Tensor::<3>::zeros([2, 3, 8], &device));

        let recorded = crate::recording::recorded(|| {
            mha.forward(input);
        });

        assert_eq!(
            recorded.lines(),
            [
                "multi head attention (unpriced)",
                "  query (unpriced)",
                "    linear",
                "  key (unpriced)",
                "    linear",
                "  value (unpriced)",
                "    linear",
                "  scaled dot product",
                "  output (unpriced)",
                "    linear",
            ]
        );
        // Two heads of four over three positions: each of the query, key and value heads is
        // `[2, 2, 3, 4]`, and the weights `[2, 2, 3, 3]`.
        let heads = 2 * 2 * 3 * 4;
        assert_eq!(
            recorded.declared("scaled dot product"),
            Some(Bounds {
                read: 3 * heads * 4,
                write: (heads + 2 * 2 * 3 * 3) * 4,
                flops: 2 * (2 * 2 * 3) * 3 * (4 + 4),
            })
        );
    }

    #[rstest]
    #[case(DType::F16)]
    #[case(DType::BF16)]
    #[case(DType::F32)]
    #[case(DType::F64)]
    fn masked_attention_weights(#[case] dtype: DType) {
        let device = crate::test_device();
        let mha = MultiHeadAttentionConfig::new(2, 2)
            .with_dropout(0.0)
            // The legacy masking value must not let masked keys contribute.
            .with_min_float(-1.0)
            .init(&device);
        let scores = Tensor::<4>::from_data(
            [[
                [[num_traits::Float::ln(3.0f32), 0.0, 100.0], [1.0, 2.0, 3.0]],
                [[num_traits::Float::ln(3.0f32), 0.0, 100.0], [1.0, 2.0, 3.0]],
            ]],
            (&device, dtype),
        );
        let mask =
            Tensor::<3, Bool>::from_data([[[false, false, true], [true, true, true]]], &device);
        let weights = mha.attn_weights(scores, None, Some(mask));
        assert_eq!(weights.dtype(), dtype);
        weights.into_data().assert_approx_eq::<f32>(
            &TensorData::from([[
                [[0.75, 0.25, 0.0], [0.0, 0.0, 0.0]],
                [[0.75, 0.25, 0.0], [0.0, 0.0, 0.0]],
            ]]),
            Tolerance::absolute(0.01),
        );
    }

    #[rstest]
    #[case(DType::F16)]
    #[case(DType::BF16)]
    #[case(DType::F32)]
    #[case(DType::F64)]
    fn combined_attention_masks(#[case] dtype: DType) {
        let device = crate::test_device();
        let mha = MultiHeadAttentionConfig::new(2, 2)
            .with_dropout(0.0)
            .init(&device);
        let scores = Tensor::<4>::zeros([2, 2, 2, 3], (&device, dtype));
        let pad = Tensor::<2, Bool>::from_data([[false, false, true], [true, true, true]], &device);
        let attn = Tensor::<3, Bool>::from_data(
            [
                [[true, false, false], [true, true, false]],
                [[false, false, false], [false, false, false]],
            ],
            &device,
        );
        let weights = mha.attn_weights(scores, Some(pad), Some(attn));
        weights.into_data().assert_eq(
            &TensorData::from([
                [
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                ],
                [
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ],
            ])
            .convert_dtype(dtype),
            true,
        );
    }

    #[test]
    fn masked_attention_below_min_float_stays_normalized() {
        let device = crate::test_device();
        let mha = MultiHeadAttentionConfig::new(1, 1)
            .with_dropout(0.0)
            .init(&device);
        let scores = Tensor::<4>::from_data([[[[-20000.0, -20000.0, 0.0]]]], &device);
        let mask = Tensor::<3, Bool>::from_data([[[false, false, true]]], &device);
        mha.attn_weights(scores, None, Some(mask))
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[[[0.5, 0.5, 0.0]]]]),
                Tolerance::default(),
            );
    }

    #[rstest]
    fn masked_attention_preserves_gradients(
        #[values(false, true)] quiet: bool,
        #[values(DType::F16, DType::BF16, DType::F32, DType::F64)] dtype: DType,
    ) {
        let device = crate::test_device().autodiff();
        let mha = MultiHeadAttentionConfig::new(1, 1)
            .with_dropout(0.0)
            .with_quiet_softmax(quiet)
            .init(&device);
        let scores =
            Tensor::<4>::from_data([[[[0.0, 0.0, 10.0], [1.0, 2.0, 3.0]]]], (&device, dtype))
                .require_grad();
        let values =
            Tensor::<4>::from_data([[[[2.0], [6.0], [100.0]]]], (&device, dtype)).require_grad();
        let mask =
            Tensor::<3, Bool>::from_data([[[false, false, true], [true, true, true]]], &device);
        let context = mha
            .attn_weights(scores.clone(), None, Some(mask))
            .matmul(values.clone());
        let weight = if quiet { 1.0f32 / 3.0 } else { 0.5 };
        let expected_context = 8.0 * weight;
        context.clone().into_data().assert_approx_eq::<f32>(
            &TensorData::from([[[[expected_context], [0.0]]]]),
            Tolerance::absolute(0.02),
        );
        let grads = context.sum().backward();
        scores
            .grad(&grads)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[[
                    [
                        (2.0 - expected_context) * weight,
                        (6.0 - expected_context) * weight,
                        0.0,
                    ],
                    [0.0, 0.0, 0.0],
                ]]]),
                Tolerance::absolute(0.01),
            );
        values
            .grad(&grads)
            .unwrap()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::from([[[[weight], [weight], [0.0]]]]),
                Tolerance::absolute(0.01),
            );
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    fn fully_masked_attention_ignores_values(
        #[case] cached: bool,
        #[values(false, true)] quiet: bool,
    ) {
        let device = crate::test_device();
        let mha = MultiHeadAttentionConfig::new(2, 1)
            .with_dropout(0.0)
            .with_output_bias(false)
            .with_quiet_softmax(quiet)
            .init(&device);
        let query = Tensor::<3>::zeros([1, 1, 2], &device);
        let key = Tensor::<3>::zeros([1, 2, 2], &device);
        let mask = Tensor::<2, Bool>::from_data([[true, true]], &device);
        for value in [1.0, 100.0] {
            let input = MhaInput::new(
                query.clone(),
                key.clone(),
                Tensor::full([1, 2, 2], value, &device),
            )
            .mask_pad(mask.clone());
            let output = if cached {
                mha.forward_cache(input, &mut MhaCache::autoregressive_cross_attention())
            } else {
                mha.forward(input)
            };
            output
                .weights
                .into_data()
                .assert_eq(&TensorData::zeros::<f32, _>([1, 1, 1, 2]), true);
            output
                .context
                .into_data()
                .assert_eq(&TensorData::zeros::<f32, _>([1, 1, 2]), true);
        }
    }

    #[test]
    fn attention_dropout_applies_after_softmax() {
        let device = Device::default().autodiff();
        device.seed(42);
        let mha = MultiHeadAttentionConfig::new(1, 1)
            .with_dropout(0.5)
            .init(&device);

        // Equal logits give probabilities of 0.5. Inverted dropout then makes
        // every returned weight either 0 or 1, regardless of the random mask.
        let scores = Tensor::<4>::ones([64, 1, 1, 2], &device).mul_scalar(-1.0);
        let weights = mha.attn_weights(scores, None, None).into_data();
        for &weight in weights.as_slice::<f32>().unwrap() {
            assert!(
                (weight - 0.0).abs() < 1e-6 || (weight - 1.0).abs() < 1e-6,
                "expected post-softmax dropout weight 0 or 1, got {weight}"
            );
        }
    }

    #[test]
    #[should_panic(
        expected = "assert_shape!(input.key, [batch_size, _, self.d_model]): axis 0 expected 2, got 1"
    )]
    fn key_batch_must_match_query_batch() {
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(8, 2).init(&device);
        let query = Tensor::zeros([2, 3, 8], &device);
        let key = Tensor::zeros([1, 3, 8], &device);
        let _ = mha.forward(MhaInput::new(query, key.clone(), key));
    }

    #[test]
    #[should_panic(
        expected = "assert_shape!(input.value, [batch_size, seq_length_2, self.d_model]): axis 0 expected 2, got 1"
    )]
    fn value_batch_must_match_query_batch() {
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(8, 2).init(&device);
        let query = Tensor::zeros([2, 3, 8], &device);
        let key = Tensor::zeros([2, 3, 8], &device);
        let value = Tensor::zeros([1, 3, 8], &device);
        let _ = mha.forward(MhaInput::new(query, key, value));
    }

    #[test]
    #[should_panic(
        expected = "assert_shape!(input.key, [batch_size, _, self.d_model]): axis 0 expected 2, got 1"
    )]
    fn cache_key_batch_must_match_query_batch() {
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(8, 2).init(&device);
        let mut cache = MhaCache::autoregressive();
        let query = Tensor::zeros([2, 3, 8], &device);
        let key = Tensor::zeros([1, 3, 8], &device);
        let _ = mha.forward_cache(MhaInput::new(query, key.clone(), key), &mut cache);
    }

    #[test]
    fn test_enable_bias() {
        let d_model = 32;
        let n_heads = 4;
        let device = Default::default();

        let base_cfg = MultiHeadAttentionConfig::new(d_model, n_heads);

        // Default, everything is enabled.
        let mha = base_cfg.init(&device);
        assert!(mha.query.bias.is_some());
        assert!(mha.key.bias.is_some());
        assert!(mha.value.bias.is_some());
        assert!(mha.output.bias.is_some());

        let mha = base_cfg.clone().with_query_bias(false).init(&device);
        assert!(mha.query.bias.is_none());
        assert!(mha.key.bias.is_some());
        assert!(mha.value.bias.is_some());
        assert!(mha.output.bias.is_some());

        let mha = base_cfg.clone().with_key_bias(false).init(&device);
        assert!(mha.query.bias.is_some());
        assert!(mha.key.bias.is_none());
        assert!(mha.value.bias.is_some());
        assert!(mha.output.bias.is_some());

        let mha = base_cfg.clone().with_value_bias(false).init(&device);
        assert!(mha.query.bias.is_some());
        assert!(mha.key.bias.is_some());
        assert!(mha.value.bias.is_none());
        assert!(mha.output.bias.is_some());

        let mha = base_cfg.clone().with_output_bias(false).init(&device);
        assert!(mha.query.bias.is_some());
        assert!(mha.key.bias.is_some());
        assert!(mha.value.bias.is_some());
        assert!(mha.output.bias.is_none());
    }

    #[test]
    fn test_self_attention_shapes() {
        let [batch_size, seq_length, d_model, n_heads] = [7, 13, 32, 4];
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);
        let input = MhaInput::self_attn(Tensor::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        ));

        let output = mha.forward(input);

        assert_eq!(
            output.context.shape(),
            Shape::new([batch_size, seq_length, d_model]),
            "Context should have the correct shape",
        );
        assert_eq!(
            output.weights.shape(),
            Shape::new([batch_size, n_heads, seq_length, seq_length]),
            "Weights should have the correct shape",
        );
    }

    #[test]
    fn test_generic_mha_shapes() {
        let [batch_size, seq_length_1, seq_length_2, d_model, n_heads] = [7, 13, 15, 32, 4];
        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init(&Default::default());
        let device = Default::default();
        let input = MhaInput::new(
            Tensor::random(
                [batch_size, seq_length_1, d_model],
                Distribution::Default,
                &device,
            ),
            Tensor::random(
                [batch_size, seq_length_2, d_model],
                Distribution::Default,
                &device,
            ),
            Tensor::random(
                [batch_size, seq_length_2, d_model],
                Distribution::Default,
                &device,
            ),
        );

        let output = mha.forward(input);

        assert_eq!(
            output.context.shape(),
            Shape::new([batch_size, seq_length_1, d_model]),
            "Context should have the correct shape",
        );
        assert_eq!(
            output.weights.shape(),
            Shape::new([batch_size, n_heads, seq_length_1, seq_length_2]),
            "Weights should have the correct shape",
        );
    }

    #[test]
    fn test_self_attention_mask_pad() {
        let [batch_size, seq_length, d_model, n_heads, num_padded] = [3, 6, 32, 2, 2];
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);

        // Create a padding mask
        let mask_pad: Tensor<2, Int> = Tensor::zeros([batch_size, seq_length], &device);
        let mask_pad = mask_pad.slice_assign(
            [0..batch_size, seq_length - num_padded..seq_length],
            Tensor::ones([batch_size, num_padded], &device),
        );
        let mask_pad = mask_pad.equal_scalar(1).to_device(&device);

        let tensor_1 = Tensor::<3>::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        // Change the end of the tensor
        let tensor_2 = tensor_1.clone().slice_assign(
            [
                0..batch_size,
                seq_length - num_padded..seq_length,
                0..d_model,
            ],
            Tensor::random(
                [batch_size, num_padded, d_model],
                Distribution::Default,
                &device,
            ),
        );

        let input_1 = MhaInput::self_attn(tensor_1).mask_pad(mask_pad.clone());
        let input_2 = MhaInput::self_attn(tensor_2).mask_pad(mask_pad);

        let output_1 = mha.forward(input_1);
        let output_2 = mha.forward(input_2);

        // Check that the beginning of each tensor is the same
        output_1
            .context
            .slice([0..batch_size, 0..seq_length - num_padded, 0..d_model])
            .into_data()
            .assert_approx_eq(
                &output_2
                    .context
                    .slice([0..batch_size, 0..seq_length - num_padded, 0..d_model])
                    .into_data(),
                Tolerance::<f32>::default(),
            );
    }

    #[test]
    fn test_autoregressive_mask_should_have_same_output_as_autoregressive_decoding() {
        let [batch_size, seq_length, d_model, n_heads] = [3, 4, 12, 2];
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init(&device);

        let tensor = Tensor::<3>::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, &tensor.device());
        let input = MhaInput::self_attn(tensor.clone()).mask_attn(mask_attn);

        let output_1 = mha.forward(input);
        let mut output_2 = Vec::new();
        let mut cache = MhaCache::autoregressive();

        for i in 1..seq_length + 1 {
            let tensor = tensor.clone().slice([0..batch_size, 0..i, 0..d_model]);
            let input = MhaInput::self_attn(tensor);
            let next_tok = mha.forward_cache(input, &mut cache).context.slice([
                0..batch_size,
                i - 1..i,
                0..d_model,
            ]);
            output_2.push(next_tok);
        }

        let output_2 = Tensor::cat(output_2, 1);

        output_1
            .context
            .into_data()
            .assert_approx_eq::<f32>(&output_2.into_data(), Tolerance::default());
    }

    #[test]
    fn display() {
        let config = MultiHeadAttentionConfig::new(2, 4);
        let mha = config.init(&Default::default());

        assert_eq!(
            alloc::format!("{mha}"),
            "MultiHeadAttention {d_model: 2, n_heads: 4, d_k: 0, \
            dropout: 0.1, min_float: -10000, quiet_softmax: false, params: 24}"
        );
    }
}
