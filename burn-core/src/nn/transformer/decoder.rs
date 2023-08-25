use alloc::vec::Vec;
use burn_tensor::Bool;

use crate::{
    self as burn,
    nn::{attention::MhaCache, cache::TensorCache, Initializer},
};

use super::{PositionWiseFeedForward, PositionWiseFeedForwardConfig};
use crate::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig,
    },
    tensor::{backend::Backend, Tensor},
};

/// Configuration to create a [Transformer Decoder](TransformerDecoder) layer.
#[derive(Config)]
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
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/libm::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// The transformer decoder module as describe in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
///
/// # Params
///
/// - layers: transformer decoder layers with `d_model` input and output features.
#[derive(Module, Debug)]
pub struct TransformerDecoder<B: Backend> {
    layers: Vec<TransformerDecoderLayer<B>>,
}

impl TransformerDecoderConfig {
    /// Initialize a new [Transformer Decoder](TransformerDecoder) module.
    pub fn init<B: Backend>(&self) -> TransformerDecoder<B> {
        let layers = (0..self.n_layers)
            .map(|_| TransformerDecoderLayer::new(self))
            .collect::<Vec<_>>();

        TransformerDecoder { layers }
    }

    /// Initialize a new [Transformer Decoder](TransformerDecoder) module with a record.
    ///
    /// # Params
    ///
    /// - record: the record to initialize the module with.
    pub fn init_with<B: Backend>(
        &self,
        record: TransformerDecoderRecord<B>,
    ) -> TransformerDecoder<B> {
        TransformerDecoder {
            layers: record
                .layers
                .into_iter()
                .map(|record| TransformerDecoderLayer::new_with(self, record))
                .collect(),
        }
    }
}

/// [Transformer Decoder](TransformerDecoder) forward pass input argument.
#[derive(Debug)]
pub struct TransformerDecoderInput<B: Backend> {
    target: Tensor<B, 3>,
    target_mask_pad: Option<Tensor<B, 2, Bool>>,
    target_mask_attn: Option<Tensor<B, 3, Bool>>,
    memory: Tensor<B, 3>,
    memory_mask_pad: Option<Tensor<B, 2, Bool>>,
    memory_mask_attn: Option<Tensor<B, 3, Bool>>,
}

impl<B: Backend> TransformerDecoderInput<B> {
    /// Create a [transformer decoder](TransformerDecoder) input argument.
    pub fn new(target: Tensor<B, 3>, memory: Tensor<B, 3>) -> Self {
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
    pub fn memory_mask_pad(mut self, mask_pad: Tensor<B, 2, Bool>) -> Self {
        self.memory_mask_pad = Some(mask_pad);
        self
    }

    /// Register the memory attention mask.
    pub fn memory_mask_attn(mut self, mask_attn: Tensor<B, 3, Bool>) -> Self {
        self.memory_mask_attn = Some(mask_attn);
        self
    }

    /// Register the target padding mask.
    pub fn target_mask_pad(mut self, mask_pad: Tensor<B, 2, Bool>) -> Self {
        self.target_mask_pad = Some(mask_pad);
        self
    }

    /// Register the target attention mask.
    pub fn target_mask_attn(mut self, mask_attn: Tensor<B, 3, Bool>) -> Self {
        self.target_mask_attn = Some(mask_attn);
        self
    }
}

/// [Transformer Decoder](TransformerDecoder) layer module.
#[derive(Module, Debug)]
pub struct TransformerDecoderLayer<B: Backend> {
    cross_attn: MultiHeadAttention<B>,
    self_attn: MultiHeadAttention<B>,
    pwff: PositionWiseFeedForward<B>,
    norm_1: LayerNorm<B>,
    norm_2: LayerNorm<B>,
    norm_3: LayerNorm<B>,
    dropout: Dropout,
    norm_first: bool,
}

struct TransformerDecoderLayerAutoregressiveCache<B: Backend> {
    cross_attn: MhaCache<B>,
    self_attn: MhaCache<B>,
    pwff: TensorCache<B, 3>,
    norm_1: TensorCache<B, 3>,
    norm_2: TensorCache<B, 3>,
    norm_3: TensorCache<B, 3>,
}

impl<B: Backend> TransformerDecoderLayerAutoregressiveCache<B> {
    fn empty() -> Self {
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
pub struct TransformerDecoderAutoregressiveCache<B: Backend> {
    layers: Vec<TransformerDecoderLayerAutoregressiveCache<B>>,
}

impl<B: Backend> TransformerDecoderAutoregressiveCache<B> {
    fn empty(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| TransformerDecoderLayerAutoregressiveCache::empty())
                .collect(),
        }
    }
}

impl<B: Backend> TransformerDecoderLayer<B> {
    fn new(config: &TransformerDecoderConfig) -> Self {
        let self_attn = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .init();

        let cross_attn = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .init();
        let norm_1 = LayerNormConfig::new(config.d_model).init();
        let norm_2 = LayerNormConfig::new(config.d_model).init();
        let norm_3 = LayerNormConfig::new(config.d_model).init();
        let dropout = DropoutConfig::new(config.dropout).init();
        let pwff = PositionWiseFeedForwardConfig::new(config.d_model, config.d_ff)
            .with_dropout(config.dropout)
            .init();

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

    fn new_with(
        config: &TransformerDecoderConfig,
        record: TransformerDecoderLayerRecord<B>,
    ) -> Self {
        let self_attn = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .init_with(record.self_attn);
        let cross_attn = MultiHeadAttentionConfig::new(config.d_model, config.n_heads)
            .with_initializer(config.initializer.clone())
            .with_dropout(config.dropout)
            .init_with(record.cross_attn);
        let norm_1 = LayerNormConfig::new(config.d_model).init_with(record.norm_1);
        let norm_2 = LayerNormConfig::new(config.d_model).init_with(record.norm_2);
        let norm_3 = LayerNormConfig::new(config.d_model).init_with(record.norm_3);
        let dropout = DropoutConfig::new(config.dropout).init();
        let pwff = PositionWiseFeedForwardConfig::new(config.d_model, config.d_ff)
            .with_dropout(config.dropout)
            .init_with(record.pwff);

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

    fn forward(&self, mut input: TransformerDecoderInput<B>) -> TransformerDecoderInput<B> {
        let mut x_0 = input.target;

        if self.norm_first {
            x_0 = self.norm_3.forward(x_0);
        }

        let mut self_attn_input = MhaInput::self_attn(x_0.clone());
        if let Some(mask_pad) = &input.target_mask_pad {
            self_attn_input = self_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.target_mask_attn {
            self_attn_input = self_attn_input.mask_attn(mask_attn.clone());
        }

        let x_1 = self.self_attn.forward(self_attn_input);
        let x_1 = self.dropout.forward(x_1.context) + x_0;
        let x_1 = self.norm_1.forward(x_1);

        let mut cross_attn_input =
            MhaInput::new(x_1.clone(), input.memory.clone(), input.memory.clone());
        if let Some(mask_pad) = &input.memory_mask_pad {
            cross_attn_input = cross_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.memory_mask_attn {
            cross_attn_input = cross_attn_input.mask_attn(mask_attn.clone());
        }

        let x_2 = self.cross_attn.forward(cross_attn_input);
        let x_2 = self.dropout.forward(x_2.context) + x_1;
        let x_2 = self.norm_2.forward(x_2);

        let x_3 = self.pwff.forward(x_2.clone());
        let mut x_3 = self.dropout.forward(x_3) + x_2;

        if !self.norm_first {
            x_3 = self.norm_3.forward(x_3)
        }

        input.target = x_3;
        input
    }

    fn forward_autoregressive_inference(
        &self,
        mut input: TransformerDecoderInput<B>,
        cache: &mut TransformerDecoderLayerAutoregressiveCache<B>,
    ) -> TransformerDecoderInput<B> {
        let mut x_0 = input.target;

        if self.norm_first {
            x_0 = cache
                .norm_3
                .forward_autoregressive(x_0, 1, |x| self.norm_3.forward(x));
        }

        let mut self_attn_input = MhaInput::self_attn(x_0.clone());
        if let Some(mask_pad) = &input.target_mask_pad {
            self_attn_input = self_attn_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.target_mask_attn {
            self_attn_input = self_attn_input.mask_attn(mask_attn.clone());
        }

        let x_1 = self
            .self_attn
            .forward_cache(self_attn_input, &mut cache.self_attn);
        let x_1 = self.dropout.forward(x_1.context) + x_0;
        let x_1 = cache
            .norm_1
            .forward_autoregressive(x_1, 1, |x| self.norm_1.forward(x));

        let mut mha_input = MhaInput::new(x_1.clone(), input.memory.clone(), input.memory.clone());
        if let Some(mask_pad) = &input.memory_mask_pad {
            mha_input = mha_input.mask_pad(mask_pad.clone());
        }
        if let Some(mask_attn) = &input.memory_mask_attn {
            mha_input = mha_input.mask_attn(mask_attn.clone());
        }

        let x_2 = self
            .cross_attn
            .forward_cache(mha_input, &mut cache.cross_attn);
        let x_2 = self.dropout.forward(x_2.context) + x_1;
        let x_2 = cache
            .norm_2
            .forward_autoregressive(x_2, 1, |x| self.norm_2.forward(x));

        let x_3 = cache
            .pwff
            .forward_autoregressive(x_2.clone(), 1, |x| self.pwff.forward(x));
        let mut x_3 = self.dropout.forward(x_3) + x_2;

        if !self.norm_first {
            x_3 = cache
                .norm_3
                .forward_autoregressive(x_3, 1, |x| self.norm_3.forward(x));
        }

        input.target = x_3;
        input
    }
}

impl<B: Backend> TransformerDecoder<B> {
    /// Applies the forward pass.
    pub fn forward(&self, mut input: TransformerDecoderInput<B>) -> Tensor<B, 3> {
        for layer in self.layers.iter() {
            input = layer.forward(input);
        }

        input.target
    }

    /// Applies the forward pass on the input using autoregressive cache.
    pub fn forward_autoregressive_inference(
        &self,
        mut input: TransformerDecoderInput<B>,
        cache: &mut TransformerDecoderAutoregressiveCache<B>,
    ) -> Tensor<B, 3> {
        for i in 0..self.layers.len() {
            let layer = self.layers.get(i).unwrap();
            let cache = cache.layers.get_mut(i).unwrap();

            input = layer.forward_autoregressive_inference(input, cache);
        }

        input.target
    }
    /// Create an empty autoregressive cache.
    pub fn new_autoregressive_cache(&self) -> TransformerDecoderAutoregressiveCache<B> {
        TransformerDecoderAutoregressiveCache::empty(self.layers.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{nn::attention::generate_autoregressive_mask, TestBackend};
    use burn_tensor::Distribution;

    #[test]
    fn test_autoregressive_norm_last() {
        let [d_model, d_ff, n_heads, num_layers] = [12, 24, 2, 3];
        TestBackend::seed(0);

        test_autoregressive(
            TransformerDecoderConfig::new(d_model, d_ff, n_heads, num_layers)
                .with_norm_first(false),
        )
    }

    #[test]
    fn test_autoregressive_norm_first() {
        let [d_model, d_ff, n_heads, num_layers] = [12, 24, 2, 3];
        TestBackend::seed(0);

        test_autoregressive(
            TransformerDecoderConfig::new(d_model, d_ff, n_heads, num_layers).with_norm_first(true),
        )
    }

    fn test_autoregressive(config: TransformerDecoderConfig) {
        let [batch_size, seq_length, d_model] = [3, 4, config.d_model];
        let transformer = config.init();

        let memory = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
        );
        let target = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_length, d_model],
            Distribution::Default,
        );
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
        output_1
            .into_data()
            .assert_approx_eq(&output_2.into_data(), 3);
    }
}
