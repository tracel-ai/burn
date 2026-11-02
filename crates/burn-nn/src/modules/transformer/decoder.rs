use crate::Initializer;
use burn_core as burn;

use alloc::vec::Vec;

use burn::config::Config;
use burn::instrument::{self, Bounds, compute};
use burn::module::{Content, DisplaySettings, Module, ModuleDisplay};
use burn::tensor::{Bool, Device, Tensor};

use crate::activation::ActivationConfig;
use crate::cache::TensorCache;
use crate::modules::bounds;
use crate::modules::role::Role;
use crate::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig,
    attention::{MhaCache, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
};

use super::{PositionWiseFeedForward, PositionWiseFeedForwardConfig};

/// Configuration to create a [Transformer Decoder](TransformerDecoder) layer using the [init function](TransformerDecoderConfig::init).
#[derive(Config, Debug)]
pub struct TransformerDecoderConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The size of the position-wise feed-forward network.
    pub d_ff: usize,
    /// The number of attention heads.
    pub n_heads: usize,
    /// The number of layers.
    pub n_layers: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
    /// Layer norm will be applied first instead of after the other modules.
    #[config(default = false)]
    pub norm_first: bool,
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
    /// The activation function used in the position-wise feed-forward network. Default: Gelu
    #[config(default = "ActivationConfig::Gelu")]
    pub activation: ActivationConfig,
    /// The epsilon value for layer normalization. Default: 1e-5
    #[config(default = 1e-5)]
    pub layer_norm_eps: f64,
}

/// The transformer decoder module as describe in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
///
/// # Params
///
/// - layers: transformer decoder layers with `d_model` input and output features.
///
/// Should be created using [TransformerDecoderConfig]
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct TransformerDecoder {
    /// Transformer decoder layers.
    pub layers: Vec<TransformerDecoderLayer>,

    /// The size of the model.
    pub d_model: usize,

    /// The size of the position-wise feed-forward network.
    pub d_ff: usize,

    /// The number of attention heads.
    pub n_heads: usize,

    /// The number of layers.
    pub n_layers: usize,

    /// The dropout rate. Default: 0.1
    pub dropout: f64,

    /// Layer norm will be applied first instead of after the other modules.
    pub norm_first: bool,

    /// Use "quiet softmax" instead of regular softmax.
    pub quiet_softmax: bool,
}

impl ModuleDisplay for TransformerDecoder {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("d_model", &self.d_model)
            .add("d_ff", &self.d_ff)
            .add("n_heads", &self.n_heads)
            .add("n_layers", &self.n_layers)
            .add("dropout", &self.dropout)
            .add("norm_first", &self.norm_first)
            .add("quiet_softmax", &self.quiet_softmax)
            .optional()
    }
}

impl TransformerDecoderConfig {
    /// Initialize a new [Transformer Decoder](TransformerDecoder) module.
    pub fn init(&self, device: &Device) -> TransformerDecoder {
        let layers = (0..self.n_layers)
            .map(|_| TransformerDecoderLayer::new(self, device))
            .collect::<Vec<_>>();

        TransformerDecoder {
            layers,
            d_model: self.d_model,
            d_ff: self.d_ff,
            n_heads: self.n_heads,
            n_layers: self.n_layers,
            dropout: self.dropout,
            norm_first: self.norm_first,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

/// [Transformer Decoder](TransformerDecoder) forward pass input argument.
#[derive(Debug)]
pub struct TransformerDecoderInput {
    target: Tensor<3>,
    target_mask_pad: Option<Tensor<2, Bool>>,
    target_mask_attn: Option<Tensor<3, Bool>>,
    memory: Tensor<3>,
    memory_mask_pad: Option<Tensor<2, Bool>>,
    memory_mask_attn: Option<Tensor<3, Bool>>,
}

impl TransformerDecoderInput {
    /// The bytes of the target, the memory and every mask: what a decoder reads of its input.
    fn bytes(&self) -> u64 {
        instrument::tensor_bytes(&self.target)
            + instrument::tensor_bytes(&self.memory)
            + bounds::masks(&self.target_mask_pad, &self.target_mask_attn)
            + bounds::masks(&self.memory_mask_pad, &self.memory_mask_attn)
    }

    /// Create a [transformer decoder](TransformerDecoder) input argument.
    pub fn new(target: Tensor<3>, memory: Tensor<3>) -> Self {
        Self {
            target,
            target_mask_pad: None,
            target_mask_attn: None,
            memory,
            memory_mask_pad: None,
            memory_mask_attn: None,
        }
    }

    /// Register the memory padding mask.
    pub fn memory_mask_pad(mut self, mask_pad: Tensor<2, Bool>) -> Self {
        self.memory_mask_pad = Some(mask_pad);
        self
    }

    /// Register the memory attention mask.
    pub fn memory_mask_attn(mut self, mask_attn: Tensor<3, Bool>) -> Self {
        self.memory_mask_attn = Some(mask_attn);
        self
    }

    /// Register the target padding mask.
    pub fn target_mask_pad(mut self, mask_pad: Tensor<2, Bool>) -> Self {
        self.target_mask_pad = Some(mask_pad);
        self
    }

    /// Register the target attention mask.
    pub fn target_mask_attn(mut self, mask_attn: Tensor<3, Bool>) -> Self {
        self.target_mask_attn = Some(mask_attn);
        self
    }
}

/// [Transformer Decoder](TransformerDecoder) layer module.
#[derive(Module, Debug)]
pub struct TransformerDecoderLayer {
    /// Cross-attention module.
    pub cross_attn: MultiHeadAttention,
    /// Self-attention module.
    pub self_attn: MultiHeadAttention,
    /// Position-wise feed-forward module.
    pub pwff: PositionWiseFeedForward,
    /// First layer norm.
    pub norm_1: LayerNorm,
    /// Second layer norm.
    pub norm_2: LayerNorm,
    /// Third layer norm.
    pub norm_3: LayerNorm,
    /// Dropout.
    pub dropout: Dropout,
    /// Whether to apply norm first.
    pub norm_first: bool,
}

/// Autoregressive cache for a single [Transformer Decoder Layer](TransformerDecoderLayer).
pub struct TransformerDecoderLayerAutoregressiveCache {
    /// Cross-attention cache.
    pub cross_attn: MhaCache,
    /// Self-attention cache.
    pub self_attn: MhaCache,
    /// Position-wise feed-forward cache.
    pub pwff: TensorCache<3>,
    /// First layer norm cache.
    pub norm_1: TensorCache<3>,
    /// Second layer norm cache.
    pub norm_2: TensorCache<3>,
    /// Third layer norm cache.
    pub norm_3: TensorCache<3>,
}

impl TransformerDecoderLayerAutoregressiveCache {
    /// Create an empty cache.
    pub fn empty() -> Self {
        Self {
            cross_attn: MhaCache::autoregressive_cross_attention(),
            self_attn: MhaCache::autoregressive(),
            pwff: TensorCache::empty(),
            norm_1: TensorCache::empty(),
            norm_2: TensorCache::empty(),
            norm_3: TensorCache::empty(),
        }
    }
}

/// Autoregressive cache for the [Transformer Decoder](TransformerDecoder) layer.
///
/// To be used during inference when decoding tokens.
pub struct TransformerDecoderAutoregressiveCache {
    layers: Vec<TransformerDecoderLayerAutoregressiveCache>,
}

impl TransformerDecoderAutoregressiveCache {
    fn empty(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| TransformerDecoderLayerAutoregressiveCache::empty())
                .collect(),
        }
    }
}

impl TransformerDecoderLayer {
    /// Create a new [TransformerDecoderLayer](TransformerDecoderLayer).
    pub fn new(config: &TransformerDecoderConfig, device: &Device) -> Self {
        let self_attn = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .with_quiet_softmax(config.quiet_softmax)
            .init(device);

        let cross_attn = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .with_quiet_softmax(config.quiet_softmax)
            .init(device);
        let norm_1 = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let norm_2 = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let norm_3 = LayerNormConfig::new(config.d_model)
            .with_epsilon(config.layer_norm_eps)
            .init(device);
        let dropout = DropoutConfig::new(config.dropout).init();
        let pwff = PositionWiseFeedForwardConfig::new(config.d_model, config.d_ff)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .with_activation(config.activation.clone())
            .init(device);

        Self {
            cross_attn,
            self_attn,
            norm_1,
            norm_2,
            norm_3,
            pwff,
            dropout,
            norm_first: config.norm_first,
        }
    }

    /// The floor for the layer alone: read its target, memory, masks and parameters once, write
    /// the target once, with its two attentions' and its feed-forward's arithmetic. Every
    /// activation between them is what a fuser keeps out of memory.
    fn bounds(&self, input: &TransformerDecoderInput) -> Bounds {
        Bounds {
            read: input.bytes() + bounds::parameters(self),
            write: instrument::tensor_bytes(&input.target),
            flops: self.flops(input),
        }
    }

    /// The layer's arithmetic over `input`: its self-attention over the target, its cross
    /// attention from the target over the memory, and its feed-forward.
    fn flops(&self, input: &TransformerDecoderInput) -> u64 {
        self.self_attn.flops(&input.target, &input.target)
            + self.cross_attn.flops(&input.target, &input.memory)
            + self.pwff.bounds(&input.target).flops
    }

    /// Applies the TransformerDecoder forward pass to the input tensor.
    #[compute(bounds = self.bounds(&input))]
    pub fn forward(&self, mut input: TransformerDecoderInput) -> TransformerDecoderInput {
        // Self attention residual path.
        let x = input.target;
        let mut residual_path = x.clone();

        // Normalize.
        if self.norm_first {
            residual_path = Role::SELF_ATTENTION_NORM.run(|| self.norm_3.forward(residual_path));
        }

        // Self attention.
        let mut self_attn_input = MhaInput::self_attn(residual_path);
        if let Some(mask_pad) = &input.target_mask_pad {
            self_attn_input = self_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.target_mask_attn {
            self_attn_input = self_attn_input.mask_attn(mask_attn.clone());
        }
        let residual_path =
            Role::SELF_ATTENTION.run(|| self.self_attn.forward(self_attn_input).context);

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Cross attention residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            Role::CROSS_ATTENTION_NORM.run(|| self.norm_1.forward(x.clone()))
        } else {
            x = Role::SELF_ATTENTION_NORM.run(|| self.norm_1.forward(x));
            x.clone()
        };

        // Cross attention.
        let mut cross_attn_input =
            MhaInput::new(residual_path, input.memory.clone(), input.memory.clone());
        if let Some(mask_pad) = &input.memory_mask_pad {
            cross_attn_input = cross_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.memory_mask_attn {
            cross_attn_input = cross_attn_input.mask_attn(mask_attn.clone());
        }
        let residual_path =
            Role::CROSS_ATTENTION.run(|| self.cross_attn.forward(cross_attn_input).context);

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Feed forward residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            Role::FEED_FORWARD_NORM.run(|| self.norm_2.forward(x.clone()))
        } else {
            x = Role::CROSS_ATTENTION_NORM.run(|| self.norm_2.forward(x));
            x.clone()
        };

        let residual_path = self.pwff.forward(residual_path);
        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Main path.
        // Normalize.
        if !self.norm_first {
            x = Role::FEED_FORWARD_NORM.run(|| self.norm_3.forward(x))
        }

        input.target = x;
        input
    }

    /// Applies the forward pass using an autoregressive cache.
    #[compute]
    pub fn forward_autoregressive_inference(
        &self,
        mut input: TransformerDecoderInput,
        cache: &mut TransformerDecoderLayerAutoregressiveCache,
    ) -> TransformerDecoderInput {
        // Self attention residual path.
        let x = input.target;
        let mut residual_path = x.clone();

        // Normalize.
        if self.norm_first {
            residual_path = cache.norm_3.forward_autoregressive(residual_path, 1, |x| {
                Role::SELF_ATTENTION_NORM.run(|| self.norm_3.forward(x))
            });
        }

        // Self attention.
        let mut self_attn_input = MhaInput::self_attn(residual_path);
        if let Some(mask_pad) = &input.target_mask_pad {
            self_attn_input = self_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.target_mask_attn {
            self_attn_input = self_attn_input.mask_attn(mask_attn.clone());
        }
        let residual_path = Role::SELF_ATTENTION.run(|| {
            self.self_attn
                .forward_cache_context(self_attn_input, &mut cache.self_attn)
        });

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Cross attention residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            cache.norm_1.forward_autoregressive(x.clone(), 1, |x| {
                Role::CROSS_ATTENTION_NORM.run(|| self.norm_1.forward(x))
            })
        } else {
            x = cache.norm_1.forward_autoregressive(x, 1, |x| {
                Role::SELF_ATTENTION_NORM.run(|| self.norm_1.forward(x))
            });
            x.clone()
        };

        // Cross attention.
        let mut cross_attn_input =
            MhaInput::new(residual_path, input.memory.clone(), input.memory.clone());
        if let Some(mask_pad) = &input.memory_mask_pad {
            cross_attn_input = cross_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.memory_mask_attn {
            cross_attn_input = cross_attn_input.mask_attn(mask_attn.clone());
        }
        let residual_path = Role::CROSS_ATTENTION.run(|| {
            self.cross_attn
                .forward_cache_context(cross_attn_input, &mut cache.cross_attn)
        });

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Feed forward residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            cache.norm_2.forward_autoregressive(x.clone(), 1, |x| {
                Role::FEED_FORWARD_NORM.run(|| self.norm_2.forward(x))
            })
        } else {
            x = cache.norm_2.forward_autoregressive(x, 1, |x| {
                Role::CROSS_ATTENTION_NORM.run(|| self.norm_2.forward(x))
            });
            x.clone()
        };

        let residual_path = cache
            .pwff
            .forward_autoregressive(residual_path, 1, |x| self.pwff.forward(x));
        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Main path.
        // Normalize.
        if !self.norm_first {
            x = cache.norm_3.forward_autoregressive(x, 1, |x| {
                Role::FEED_FORWARD_NORM.run(|| self.norm_3.forward(x))
            })
        }

        input.target = x;
        input
    }
}

impl TransformerDecoder {
    /// The floor for the decoder alone: read its target, memory, masks and every layer's
    /// parameters once, write the target once, with every layer's arithmetic.
    fn bounds(&self, input: &TransformerDecoderInput) -> Bounds {
        Bounds {
            read: input.bytes() + bounds::parameters(self),
            write: instrument::tensor_bytes(&input.target),
            flops: self.layers.iter().map(|layer| layer.flops(input)).sum(),
        }
    }

    /// Applies the forward pass.
    #[compute(bounds = self.bounds(&input))]
    pub fn forward(&self, mut input: TransformerDecoderInput) -> Tensor<3> {
        for layer in self.layers.iter() {
            input = layer.forward(input);
        }

        input.target
    }

    /// Applies the forward pass on the input using autoregressive cache.
    #[compute]
    pub fn forward_autoregressive_inference(
        &self,
        mut input: TransformerDecoderInput,
        cache: &mut TransformerDecoderAutoregressiveCache,
    ) -> Tensor<3> {
        for i in 0..self.layers.len() {
            let layer = self.layers.get(i).unwrap();
            let cache = cache.layers.get_mut(i).unwrap();

            input = layer.forward_autoregressive_inference(input, cache);
        }

        input.target
    }
    /// Create an empty autoregressive cache.
    pub fn new_autoregressive_cache(&self) -> TransformerDecoderAutoregressiveCache {
        TransformerDecoderAutoregressiveCache::empty(self.layers.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::generate_autoregressive_mask;

    use burn::tensor::Tolerance;
    type FT = f32;

    /// A decoder moves no more than its parts do, in either direction, and does their arithmetic:
    /// what it keeps out of memory is the traffic between them a fuser could remove.
    #[cfg(feature = "std")]
    #[test]
    fn a_decoder_moves_within_its_parts() {
        let device = Device::default();
        let decoder = TransformerDecoderConfig::new(8, 16, 2, 2).init(&device);
        let input = || {
            TransformerDecoderInput::new(
                Tensor::<3>::zeros([1, 3, 8], &device),
                Tensor::<3>::zeros([1, 5, 8], &device),
            )
        };
        let layer = &decoder.layers[0];

        for recorded in [
            crate::recording::recorded(|| {
                decoder.forward(input());
            }),
            crate::recording::recorded(|| {
                layer.forward(input());
            }),
        ] {
            let declared = recorded.lines()[0].clone();
            let whole = recorded
                .declared(&declared)
                .expect("the composite is priced");
            let parts = recorded.parts();
            assert!(
                whole.moves_within(parts),
                "{declared}: {whole:?} past {parts:?}"
            );
            assert!(
                whole.bytes() < parts.bytes(),
                "{declared} keeps traffic out of memory"
            );
            assert_eq!(whole.flops, parts.flops, "{declared}");
        }
    }

    /// A decoder layer's self and cross attention are one type and stay two components, as do
    /// its three norms, each named for the sublayer it normalizes around: before it with
    /// `norm_first`, after it otherwise.
    #[cfg(feature = "std")]
    #[test]
    fn a_decoder_layer_keeps_its_attentions_apart() {
        let device = Device::default();
        let post_norm = [
            "self attention (unpriced)",
            "self attention norm (unpriced)",
            "cross attention (unpriced)",
            "cross attention norm (unpriced)",
            "position wise feed forward",
            "feed forward norm (unpriced)",
        ];
        let pre_norm = [
            "self attention norm (unpriced)",
            "self attention (unpriced)",
            "cross attention norm (unpriced)",
            "cross attention (unpriced)",
            "feed forward norm (unpriced)",
            "position wise feed forward",
        ];

        for (norm_first, expected) in [(false, post_norm), (true, pre_norm)] {
            let decoder = TransformerDecoderConfig::new(8, 16, 2, 1)
                .with_norm_first(norm_first)
                .init(&device);
            let input = TransformerDecoderInput::new(
                Tensor::<3>::zeros([1, 3, 8], &device),
                Tensor::<3>::zeros([1, 5, 8], &device),
            );

            let recorded = crate::recording::recorded(|| {
                decoder.forward(input);
            });

            // The layer's sublayers, two levels below the decoder.
            let sublayers: Vec<String> = recorded
                .lines()
                .into_iter()
                .filter_map(|line| line.strip_prefix("    ").map(str::to_string))
                .filter(|line| !line.starts_with(' '))
                .collect();
            assert_eq!(sublayers, expected, "norm_first: {norm_first}");
        }
    }

    #[test]
    fn test_autoregressive_norm_last() {
        let [d_model, d_ff, n_heads, num_layers] = [12, 24, 2, 3];
        let device = Device::default();
        device.seed(0);

        test_autoregressive(
            TransformerDecoderConfig::new(d_model, d_ff, n_heads, num_layers)
                .with_norm_first(false),
        )
    }

    #[test]
    fn test_autoregressive_norm_first() {
        let [d_model, d_ff, n_heads, num_layers] = [12, 24, 2, 3];
        let device = Device::default();
        device.seed(0);

        test_autoregressive(
            TransformerDecoderConfig::new(d_model, d_ff, n_heads, num_layers).with_norm_first(true),
        )
    }

    fn test_autoregressive(config: TransformerDecoderConfig) {
        let device = Default::default();
        let [batch_size, seq_length, d_model] = [3, 4, config.d_model];
        let transformer = config.init(&device);

        let memory = Tensor::arange(0..(batch_size * seq_length * d_model) as i64, &device)
            .float()
            .reshape([batch_size, seq_length, d_model]);
        let target = Tensor::arange(0..(batch_size * seq_length * d_model) as i64, &device)
            .float()
            .reshape([batch_size, seq_length, d_model]);
        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, &target.device());
        let input = TransformerDecoderInput::new(target.clone(), memory.clone())
            .target_mask_attn(mask_attn);

        // Normal forward using masking.
        let output_1 = transformer.forward(input);

        // Forward using the autoregressive cache.
        let mut output_2 = Vec::new();
        let mut cache = transformer.new_autoregressive_cache();

        for i in 1..seq_length + 1 {
            let target = target.clone().slice([0..batch_size, 0..i, 0..d_model]);

            let mask_attn = generate_autoregressive_mask(batch_size, i, &target.device());
            let input = TransformerDecoderInput::new(target.clone(), memory.clone())
                .target_mask_attn(mask_attn);
            let next_tok = transformer // Greedy sampling
                .forward_autoregressive_inference(input, &mut cache)
                .slice([0..batch_size, i - 1..i, 0..d_model]);
            output_2.push(next_tok);
        }

        let output_2 = Tensor::cat(output_2, 1);

        // Should produce the same tokens.
        let tolerance = Tolerance::rel_abs(5e-3, 1e-4);
        output_1
            .into_data()
            .assert_approx_eq::<FT>(&output_2.into_data(), tolerance);
    }

    #[test]
    fn display() {
        let config = TransformerDecoderConfig::new(2, 4, 2, 3);
        let transformer = config.init(&Default::default());

        assert_eq!(
            alloc::format!("{transformer}"),
            "TransformerDecoder {d_model: 2, d_ff: 4, n_heads: 2, n_layers: 3, \
            dropout: 0.1, norm_first: false, quiet_softmax: false, params: 246}"
        );
    }
}
