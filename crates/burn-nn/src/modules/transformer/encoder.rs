use burn_core as burn;

use alloc::vec::Vec;

use super::{PositionWiseFeedForward, PositionWiseFeedForwardConfig};
use crate::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig,
    attention::{MhaCache, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    cache::TensorCache,
};
use burn::config::Config;
use burn::module::{Content, DisplaySettings, Initializer, Module, ModuleDisplay};
use burn::tensor::{Bool, Tensor, backend::Backend};

/// Configuration to create a [Transformer Encoder](TransformerEncoder) layer using the [init function](TransformerEncoderConfig::init).
#[derive(Config, Debug)]
pub struct TransformerEncoderConfig {
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
}

/// The transformer encoder module as describe in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
///
/// # Params
///
/// - layers: transformer encoder layers with `d_model` input and output features.
///
/// Should be created using [TransformerEncoderConfig]
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct TransformerEncoder<B: Backend> {
    /// The transformer encoder layers.
    pub layers: Vec<TransformerEncoderLayer<B>>,

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

impl<B: Backend> ModuleDisplay for TransformerEncoder<B> {
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

/// [Transformer Encoder](TransformerEncoder) forward pass input argument.
#[derive(Debug)]
pub struct TransformerEncoderInput<B: Backend> {
    tensor: Tensor<B, 3>,
    mask_pad: Option<Tensor<B, 2, Bool>>,
    mask_attn: Option<Tensor<B, 3, Bool>>,
}

impl<B: Backend> TransformerEncoderInput<B> {
    /// Create a [transformer encoder](TransformerEncoder) input argument.
    pub fn new(tensor: Tensor<B, 3>) -> Self {
        Self {
            tensor,
            mask_pad: None,
            mask_attn: None,
        }
    }

    /// Register the padding mask.
    pub fn mask_pad(mut self, mask_pad: Tensor<B, 2, Bool>) -> Self {
        self.mask_pad = Some(mask_pad);
        self
    }

    /// Register the attention mask.
    pub fn mask_attn(mut self, mask_attn: Tensor<B, 3, Bool>) -> Self {
        self.mask_attn = Some(mask_attn);
        self
    }
}
impl TransformerEncoderConfig {
    /// Initialize a new [transformer encoder](TransformerEncoder) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TransformerEncoder<B> {
        let layers = (0..self.n_layers)
            .map(|_| TransformerEncoderLayer::new(self, device))
            .collect::<Vec<_>>();

        TransformerEncoder {
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

impl<B: Backend> TransformerEncoder<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - tensor: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(&self, input: TransformerEncoderInput<B>) -> Tensor<B, 3> {
        let mut x = input.tensor;

        for layer in self.layers.iter() {
            x = layer.forward(x, input.mask_pad.clone(), input.mask_attn.clone());
        }

        x
    }
    /// Applies the forward pass on the input tensor using autoregressive cache.
    ///
    /// # Shapes
    ///
    /// - tensor: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward_autoregressive_inference(
        &self,
        input: TransformerEncoderInput<B>,
        cache: &mut TransformerEncoderAutoregressiveCache<B>,
    ) -> Tensor<B, 3> {
        let mut x = input.tensor;

        for i in 0..self.layers.len() {
            let layer = self.layers.get(i).unwrap();
            let cache = cache.layers.get_mut(i).unwrap();

            x = layer.forward_autoregressive_inference(
                x,
                input.mask_pad.clone(),
                input.mask_attn.clone(),
                cache,
            );
        }

        x
    }

    /// Create an empty autoregressive cache.
    pub fn new_autoregressive_cache(&self) -> TransformerEncoderAutoregressiveCache<B> {
        TransformerEncoderAutoregressiveCache::empty(self.layers.len())
    }
}

/// Transformer encoder layer module.
#[derive(Module, Debug)]
pub struct TransformerEncoderLayer<B: Backend> {
    pub mha: MultiHeadAttention<B>,
    pub pwff: PositionWiseFeedForward<B>,
    pub norm_1: LayerNorm<B>,
    pub norm_2: LayerNorm<B>,
    pub dropout: Dropout,
    pub norm_first: bool,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    fn new(config: &TransformerEncoderConfig, device: &B::Device) -> Self {
        let mha = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .with_quiet_softmax(config.quiet_softmax)
            .init(device);
        let norm_1 = LayerNormConfig::new(config.d_model).init(device);
        let norm_2 = LayerNormConfig::new(config.d_model).init(device);
        let dropout = DropoutConfig::new(config.dropout).init();
        let pwff = PositionWiseFeedForwardConfig::new(config.d_model, config.d_ff)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .init(device);

        Self {
            mha,
            norm_1,
            norm_2,
            pwff,
            dropout,
            norm_first: config.norm_first,
        }
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
        mask_attn: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 3> {
        // Multi-head attention residual path.
        let x = input;
        let mut residual_path = x.clone();

        // Normalize.
        if self.norm_first {
            residual_path = self.norm_2.forward(residual_path)
        }

        // Multi-head attention.
        let mut input_mhs = MhaInput::self_attn(residual_path);
        if let Some(mask_pad) = mask_pad {
            input_mhs = input_mhs.mask_pad(mask_pad);
        }
        if let Some(mask_attn) = mask_attn {
            input_mhs = input_mhs.mask_attn(mask_attn);
        }
        let residual_path = self.mha.forward(input_mhs).context;

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Feed forward residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            self.norm_1.forward(x.clone())
        } else {
            x = self.norm_1.forward(x);
            x.clone()
        };

        // Feed forward.
        let residual_path = self.pwff.forward(residual_path);
        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Main path.
        // Normalize.
        if !self.norm_first {
            x = self.norm_2.forward(x)
        }

        x
    }

    fn forward_autoregressive_inference(
        &self,
        input: Tensor<B, 3>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
        mask_attn: Option<Tensor<B, 3, Bool>>,
        cache: &mut TransformerEncoderLayerAutoregressiveCache<B>,
    ) -> Tensor<B, 3> {
        // Multi-head attention residual path.
        let x = input;
        let mut residual_path = x.clone();

        // Normalize.
        if self.norm_first {
            residual_path = cache
                .norm_2
                .forward_autoregressive(residual_path, 1, |x| self.norm_2.forward(x))
        }

        // Multi-head attention.
        let mut input_mhs = MhaInput::self_attn(residual_path);
        if let Some(mask_pad) = mask_pad {
            input_mhs = input_mhs.mask_pad(mask_pad);
        }
        if let Some(mask_attn) = mask_attn {
            input_mhs = input_mhs.mask_attn(mask_attn);
        }
        let residual_path = self.mha.forward_cache(input_mhs, &mut cache.mha).context;

        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Feed forward residual path.
        // Normalize.
        let residual_path = if self.norm_first {
            cache
                .norm_1
                .forward_autoregressive(x.clone(), 1, |x| self.norm_1.forward(x))
        } else {
            x = cache
                .norm_1
                .forward_autoregressive(x, 1, |x| self.norm_1.forward(x));
            x.clone()
        };

        // Feed forward.
        let residual_path = cache
            .pwff
            .forward_autoregressive(residual_path, 1, |x| self.pwff.forward(x));
        let residual_path = self.dropout.forward(residual_path);
        let mut x = x + residual_path;

        // Main path.
        // Normalize.
        if !self.norm_first {
            x = cache
                .norm_2
                .forward_autoregressive(x, 1, |x| self.norm_2.forward(x))
        }

        x
    }
}

struct TransformerEncoderLayerAutoregressiveCache<B: Backend> {
    mha: MhaCache<B>,
    pwff: TensorCache<B, 3>,
    norm_1: TensorCache<B, 3>,
    norm_2: TensorCache<B, 3>,
}

impl<B: Backend> TransformerEncoderLayerAutoregressiveCache<B> {
    fn empty() -> Self {
        Self {
            mha: MhaCache::autoregressive(),
            pwff: TensorCache::empty(),
            norm_1: TensorCache::empty(),
            norm_2: TensorCache::empty(),
        }
    }
}

/// Autoregressive cache for the [Transformer Encoder](TransformerEncoder) layer.
///
/// To be used during inference when decoding tokens.
pub struct TransformerEncoderAutoregressiveCache<B: Backend> {
    layers: Vec<TransformerEncoderLayerAutoregressiveCache<B>>,
}

impl<B: Backend> TransformerEncoderAutoregressiveCache<B> {
    fn empty(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| TransformerEncoderLayerAutoregressiveCache::empty())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestBackend, attention::generate_autoregressive_mask};
    use burn::tensor::Distribution;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_autoregressive_norm_last() {
        let [d_model, d_ff, n_heads, num_layers] = [12, 24, 2, 3];
        test_autoregressive(
            TransformerEncoderConfig::new(d_model, d_ff, n_heads, num_layers)
                .with_norm_first(false),
        )
    }

    #[test]
    fn test_autoregressive_norm_first() {
        let [d_model, d_ff, n_heads, num_layers] = [12, 24, 2, 3];
        test_autoregressive(
            TransformerEncoderConfig::new(d_model, d_ff, n_heads, num_layers).with_norm_first(true),
        )
    }

    fn test_autoregressive(config: TransformerEncoderConfig) {
        let [batch_size, seq_length, d_model] = [3, 4, config.d_model];
        let device = Default::default();
        let transformer = config.init(&device);

        let tensor = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
            &device,
        );
        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, &tensor.device());
        let input = TransformerEncoderInput::new(tensor.clone()).mask_attn(mask_attn);

        let output_1 = transformer.forward(input);
        let mut output_2 = Vec::new();
        let mut cache = transformer.new_autoregressive_cache();

        for i in 1..seq_length + 1 {
            let tensor = tensor.clone().slice([0..batch_size, 0..i, 0..d_model]);
            let input = TransformerEncoderInput::new(tensor.clone());
            let next_tok = transformer
                .forward_autoregressive_inference(input, &mut cache)
                .slice([0..batch_size, i - 1..i, 0..d_model]);
            output_2.push(next_tok);
        }

        let output_2 = Tensor::cat(output_2, 1);

        output_1
            .into_data()
            .assert_approx_eq::<FT>(&output_2.into_data(), Tolerance::permissive());
    }

    #[test]
    fn display() {
        let config = TransformerEncoderConfig::new(2, 4, 2, 3);
        let transformer = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{transformer}"),
            "TransformerEncoder {d_model: 2, d_ff: 4, n_heads: 2, \
            n_layers: 3, dropout: 0.1, norm_first: false, quiet_softmax: false, params: 162}"
        );
    }
}
