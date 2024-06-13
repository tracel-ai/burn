use crate as burn;

use crate::nn::cache::TensorCache;
use crate::nn::Initializer;
use crate::{
    config::Config,
    module::Module,
    nn,
    tensor::{activation, backend::Backend, Bool, Tensor},
};

#[cfg(not(feature = "std"))]
use num_traits::Float;

/// Configuration to create a [Multi Head Attention](MultiHeadAttention) layer using the [init function](MultiHeadAttentionConfig::init).
#[derive(Config)]
pub struct MultiHeadAttentionConfig {
    /// The size of each linear layer.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
    /// The minimum value a float can take. Default: -1.0e4
    /// This is used to mask attention scores before calculating attention weights.
    /// A value too low might result in NaN.
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
}

/// The multihead attention module as describe in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
///
/// # Params
///
/// - query: [Linear](nn::Linear) layer with `d_model` input and output features.
/// - key: [Linear](nn::Linear) layer with `d_model` input and output features.
/// - value: [Linear](nn::Linear) layer with `d_model` input and output features.
/// - output: [Linear](nn::Linear) layer with `d_model` input and output features.
///
/// Should be created with [MultiHeadAttentionConfig].
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    output: nn::Linear<B>,
    dropout: nn::Dropout,
    activation: nn::Gelu,
    n_heads: usize,
    d_k: usize,
    min_float: f64,
    quiet_softmax: bool,
}

/// [Multihead attention](MultiHeadAttention) forward pass input argument.
#[derive(Debug, Clone)]
pub struct MhaInput<B: Backend> {
    /// Shape `[batch_size, seq_length_1, d_model]`
    query: Tensor<B, 3>,
    /// Shape `[batch_size, seq_length_2, d_model]`
    key: Tensor<B, 3>,
    /// Shape `[batch_size, seq_length_2, d_model]`
    value: Tensor<B, 3>,
    mask_pad: Option<Tensor<B, 2, Bool>>,
    mask_attn: Option<Tensor<B, 3, Bool>>,
}

impl MultiHeadAttentionConfig {
    /// Initialize a new [multihead attention](MultiHeadAttention) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadAttention<B> {
        let linear = |config: &Self| {
            nn::LinearConfig::new(config.d_model, config.d_model)
                .with_initializer(self.initializer.clone())
                .init(device)
        };

        MultiHeadAttention {
            query: linear(self),
            key: linear(self),
            value: linear(self),
            output: linear(self),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
            activation: nn::Gelu::new(),
            n_heads: self.n_heads,
            d_k: self.d_model / self.n_heads,
            min_float: self.min_float,
            quiet_softmax: self.quiet_softmax,
        }
    }
}

impl<B: Backend> MhaInput<B> {
    /// Create a [multihead attention](MultiHeadAttention) input argument
    /// by setting the query, key and value to the given tensor.
    ///
    /// # Shape
    /// - tensor: `[batch_size, seq_length, d_model]`
    pub fn self_attn(tensor: Tensor<B, 3>) -> Self {
        Self {
            query: tensor.clone(),
            key: tensor.clone(),
            value: tensor,
            mask_pad: None,
            mask_attn: None,
        }
    }

    /// Create a [multihead attention](MultiHeadAttention) input argument.
    pub fn new(query: Tensor<B, 3>, key: Tensor<B, 3>, value: Tensor<B, 3>) -> Self {
        Self {
            query,
            key,
            value,
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

/// [Multihead attention](MultiHeadAttention) outputs.
#[derive(Debug, Clone)]
pub struct MhaOutput<B: Backend> {
    /// The attention weights `[batch_size, n_heads, seq_length_1, seq_length_2]`.
    pub weights: Tensor<B, 4>,
    /// The context tensor `[batch_size, seq_length_1, d_model]`.
    pub context: Tensor<B, 3>,
}

impl<B: Backend> MultiHeadAttention<B> {
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
    pub fn forward(&self, input: MhaInput<B>) -> MhaOutput<B> {
        let [batch_size, seq_length_1, d_model] = input.query.dims();

        let query = self.attention_linear(input.query, &self.query);
        let key = self.attention_linear(input.key, &self.key);
        let value = self.attention_linear(input.value, &self.value);

        let attn_scores = self.attn_scores(query, key);
        let weights = self.attn_weights(attn_scores, input.mask_pad, input.mask_attn);

        let context = weights.clone().matmul(value);
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length_1, d_model]);
        let context = self.output.forward(context);

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
    pub fn forward_cache(&self, input: MhaInput<B>, cache: &mut MhaCache<B>) -> MhaOutput<B> {
        let [batch_size, seq_length_1, d_model] = input.query.dims();

        let query = cache
            .query
            .forward(input.query, |t| self.attention_linear(t, &self.query));
        let key = cache
            .key
            .forward(input.key, |t| self.attention_linear(t, &self.key));
        let value = cache
            .value
            .forward(input.value, |t| self.attention_linear(t, &self.value));

        let attn_scores = self.attn_scores(query, key);
        let weights = self.attn_weights(attn_scores, input.mask_pad, input.mask_attn);

        let context = weights.clone().matmul(value);
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length_1, d_model]);

        let context = cache.output.forward(context, |t| self.output.forward(t));

        MhaOutput { weights, context }
    }

    fn attn_scores(&self, query: Tensor<B, 4>, key: Tensor<B, 4>) -> Tensor<B, 4> {
        let attn_scores = query
            .matmul(key.transpose())
            .div_scalar((self.d_k as f32).sqrt());

        self.dropout.forward(attn_scores)
    }

    fn attn_weights(
        &self,
        mut attn_scores: Tensor<B, 4>,
        mask_pad: Option<Tensor<B, 2, Bool>>,
        mask_attn: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 4> {
        if let Some(mask_pad) = mask_pad {
            let [batch_size, seq_length] = mask_pad.dims();

            attn_scores = attn_scores.mask_fill(
                mask_pad.reshape([batch_size, 1, 1, seq_length]),
                self.min_float,
            );
        }

        if let Some(mask_attn) = mask_attn {
            let [batch_size, seq_length_1, seq_length_2] = mask_attn.dims();

            attn_scores = attn_scores.mask_fill(
                mask_attn.reshape([batch_size, 1, seq_length_1, seq_length_2]),
                self.min_float,
            );
        }

        if self.quiet_softmax {
            activation::quiet_softmax(attn_scores, 3)
        } else {
            activation::softmax(attn_scores, 3)
        }
    }

    fn attention_linear(&self, x: Tensor<B, 3>, linear: &nn::Linear<B>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}

/// Cache for the [Multi Head Attention](MultiHeadAttention) layer.
///
/// To be used during inference when decoding tokens.
pub struct MhaCache<B: Backend> {
    query: MhaLinearCache<B, 4>,
    key: MhaLinearCache<B, 4>,
    value: MhaLinearCache<B, 4>,
    output: MhaLinearCache<B, 3>,
}

enum MhaLinearCache<B: Backend, const D: usize> {
    Autoregressive(TensorCache<B, D>, usize),
    Full(TensorCache<B, D>),
}

impl<B: Backend> MhaCache<B> {
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

impl<B: Backend, const D: usize> MhaLinearCache<B, D> {
    pub fn forward<F: Fn(Tensor<B, 3>) -> Tensor<B, D>>(
        &mut self,
        tensor: Tensor<B, 3>,
        func: F,
    ) -> Tensor<B, D> {
        match self {
            MhaLinearCache::Autoregressive(cache, dim) => {
                cache.forward_autoregressive(tensor, *dim, func)
            }
            MhaLinearCache::Full(cache) => cache.forward_full(tensor, func),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Int;
    use crate::tensor::{Distribution, Shape};
    use crate::{nn::attention::generate_autoregressive_mask, TestBackend};
    use alloc::vec::Vec;

    #[test]
    fn test_self_attention_shapes() {
        let [batch_size, seq_length, d_model, n_heads] = [7, 13, 32, 4];
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init::<TestBackend>(&device);
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
        let mha = MultiHeadAttentionConfig::new(d_model, n_heads)
            .init::<TestBackend>(&Default::default());
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
        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init::<TestBackend>(&device);

        // Create a padding mask
        let mask_pad: Tensor<TestBackend, 2, Int> =
            Tensor::zeros([batch_size, seq_length], &device);
        let mask_pad = mask_pad.slice_assign(
            [0..batch_size, seq_length - num_padded..seq_length],
            Tensor::ones([batch_size, num_padded], &device),
        );
        let mask_pad = mask_pad.equal_elem(1).to_device(&device);

        let tensor_1 = Tensor::<TestBackend, 3>::random(
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
                3,
            );
    }

    #[test]
    fn test_autoregressive_mask_should_have_same_output_as_autoregressive_decoding() {
        let [batch_size, seq_length, d_model, n_heads] = [3, 4, 12, 2];
        let device = Default::default();
        let mha = MultiHeadAttentionConfig::new(d_model, n_heads).init::<TestBackend>(&device);

        let tensor = Tensor::<TestBackend, 3>::random(
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
            .assert_approx_eq(&output_2.into_data(), 3);
    }
}
