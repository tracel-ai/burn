use crate as burn;

use super::{PositionWiseFeedForward, PositionWiseFeedForwardConfig};
use crate::{
    config::Config,
    module::{Module, Param},
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig,
    },
    tensor::{backend::Backend, BoolTensor, Tensor},
};

/// Configuration to create a [Transformer Encoder](TransformerEncoder) layer.
#[derive(Config)]
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
}

/// The transformer encoder module as describe in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
///
/// # Params
///
/// - layers: transformer encoder layers with `d_model` input and output features.
#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    layers: Param<Vec<TransformerEncoderLayer<B>>>,
}

/// [Transformer Encoder](TransformerEncoder) forward pass input argument.
#[derive(Debug)]
pub struct TransformerEncoderInput<B: Backend> {
    tensor: Tensor<B, 3>,
    mask_pad: Option<BoolTensor<B, 2>>,
    mask_attn: Option<BoolTensor<B, 3>>,
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
    pub fn mask_pad(mut self, mask_pad: BoolTensor<B, 2>) -> Self {
        self.mask_pad = Some(mask_pad);
        self
    }

    /// Register the attention mask.
    pub fn mask_attn(mut self, mask_attn: BoolTensor<B, 3>) -> Self {
        self.mask_attn = Some(mask_attn);
        self
    }
}

impl<B: Backend> TransformerEncoder<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &TransformerEncoderConfig) -> Self {
        let layers = (0..config.n_layers)
            .into_iter()
            .map(|_| TransformerEncoderLayer::new(config))
            .collect();

        Self {
            layers: Param::new(layers),
        }
    }

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
}

#[derive(Module, Debug)]
struct TransformerEncoderLayer<B: Backend> {
    mha: Param<MultiHeadAttention<B>>,
    pwff: Param<PositionWiseFeedForward<B>>,
    norm_1: Param<LayerNorm<B>>,
    norm_2: Param<LayerNorm<B>>,
    dropout: Dropout,
}

impl<B: Backend> TransformerEncoderLayer<B> {
    fn new(config: &TransformerEncoderConfig) -> Self {
        let config_norm = LayerNormConfig::new(config.d_model);
        let config_dropout = DropoutConfig::new(config.dropout);
        let config_mha = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_dropout(config.dropout);
        let config_pwff = PositionWiseFeedForwardConfig::new(config.d_model, config.d_ff)
            .with_dropout(config.dropout);

        let mha = MultiHeadAttention::new(&config_mha);
        let norm_1 = LayerNorm::new(&config_norm);
        let norm_2 = LayerNorm::new(&config_norm);
        let dropout = Dropout::new(&config_dropout);
        let pwff = PositionWiseFeedForward::new(&config_pwff);

        Self {
            mha: Param::new(mha),
            norm_1: Param::new(norm_1),
            norm_2: Param::new(norm_2),
            pwff: Param::new(pwff),
            dropout,
        }
    }

    fn forward(
        &self,
        input: Tensor<B, 3>,
        mask_pad: Option<BoolTensor<B, 2>>,
        mask_attn: Option<BoolTensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let mut input_mhs = MhaInput::self_attn(input.clone());

        if let Some(mask_pad) = mask_pad {
            input_mhs = input_mhs.mask_pad(mask_pad);
        }

        if let Some(mask_attn) = mask_attn {
            input_mhs = input_mhs.mask_attn(mask_attn);
        }

        let x_1 = self.mha.forward(input_mhs);
        let x_1 = self.dropout.forward(x_1.context) + input;
        let x_1 = self.norm_1.forward(x_1);

        let x_2 = self.pwff.forward(x_1.clone());
        let x_2 = self.dropout.forward(x_2) + x_1;

        self.norm_2.forward(x_2)
    }
}
