use crate as burn;

use crate::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{activation, backend::Backend, BoolTensor, Tensor},
};

/// Configuration to create a [Multi Head Attention](MultiHeadAttention) layer.
#[derive(Config)]
pub struct MultiHeadAttentionConfig {
    /// The size of the each linear layer.
    d_model: usize,
    /// The number of heads.
    n_heads: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    dropout: f64,
    /// The minimum value a float can take. Default: -1.0e4
    /// This is used to mask attention scores before calculating attention weights.
    /// A value too low might result in NaN.
    #[config(default = -1.0e4)]
    min_float: f64,
}

/// The multihead attention module as describe in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
///
/// # Params
///
/// - query: [Linear](nn::Linear) layer with `d_model` input and output features.
/// - key: [Linear](nn::Linear) layer with `d_model` input and output features.
/// - value: [Linear](nn::Linear) layer with `d_model` input and output features.
/// - output: [Linear](nn::Linear) layer with `d_model` input and output features.
#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    query: Param<nn::Linear<B>>,
    key: Param<nn::Linear<B>>,
    value: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
    dropout: nn::Dropout,
    activation: nn::GELU,
    n_heads: usize,
    d_k: usize,
    min_float: f64,
}

/// [Multihead attention](MultiHeadAttention) forward pass input argument.
#[derive(Debug)]
pub struct MhaInput<B: Backend> {
    query: Tensor<B, 3>,
    key: Tensor<B, 3>,
    value: Tensor<B, 3>,
    mask_pad: Option<BoolTensor<B, 2>>,
    mask_attn: Option<BoolTensor<B, 3>>,
}

impl<B: Backend> MhaInput<B> {
    /// Create a [multihead attention](MultiHeadAttention) input argument
    /// by setting the query, key and value to the given tensor.
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

/// [Multihead attention](MultiHeadAttention) outputs.
#[derive(Debug)]
pub struct MhaOutput<B: Backend> {
    /// The attention weights [batch_size, seq_length_1, seq_length_2].
    pub weights: Tensor<B, 4>,
    /// The context tensor [batch_size, seq_length_1, d_model].
    pub context: Tensor<B, 3>,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &MultiHeadAttentionConfig) -> Self {
        let linear = |config: &MultiHeadAttentionConfig| {
            Param::new(nn::Linear::new(&nn::LinearConfig::new(
                config.d_model,
                config.d_model,
            )))
        };

        Self {
            query: linear(config),
            key: linear(config),
            value: linear(config),
            output: linear(config),
            dropout: nn::Dropout::new(&nn::DropoutConfig::new(config.dropout)),
            activation: nn::GELU::new(),
            n_heads: config.n_heads,
            d_k: config.d_model / config.n_heads,
            min_float: config.min_float,
        }
    }

    /// Applies the forward pass on the input tensors.
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

        let context = weights.matmul(&value);
        let context = context
            .swap_dims(1, 2)
            .reshape([batch_size, seq_length_1, d_model]);
        let context = self.output.forward(context);

        MhaOutput { weights, context }
    }

    fn attn_scores(&self, query: Tensor<B, 4>, key: Tensor<B, 4>) -> Tensor<B, 4> {
        let attn_scores = query
            .matmul(&key.transpose())
            .div_scalar((self.d_k as f32).sqrt());

        self.dropout.forward(attn_scores)
    }

    fn attn_weights(
        &self,
        mut attn_scores: Tensor<B, 4>,
        mask_pad: Option<BoolTensor<B, 2>>,
        mask_attn: Option<BoolTensor<B, 3>>,
    ) -> Tensor<B, 4> {
        if let Some(mask_pad) = mask_pad {
            let [batch_size, seq_length] = mask_pad.dims();

            attn_scores = attn_scores.mask_fill(
                &mask_pad.reshape([batch_size, 1, 1, seq_length]),
                self.min_float,
            );
        }

        if let Some(mask_attn) = mask_attn {
            let [batch_size, seq_length_1, seq_length_2] = mask_attn.dims();

            attn_scores = attn_scores.mask_fill(
                &mask_attn.reshape([batch_size, 1, seq_length_1, seq_length_2]),
                self.min_float,
            );
        }

        activation::softmax(&attn_scores, 3)
    }

    fn attention_linear(&self, x: Tensor<B, 3>, linear: &Param<nn::Linear<B>>) -> Tensor<B, 4> {
        let [batch_size, seq_length, _d_model] = x.dims();
        linear
            .forward(x)
            .reshape([batch_size, seq_length, self.n_heads, self.d_k])
            .swap_dims(1, 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::{Distribution, Shape};

    #[test]
    fn test_self_attention_shapes() {
        let [batch_size, seq_length, d_model, n_heads] = [7, 13, 32, 4];
        let mha = MultiHeadAttention::<TestBackend>::new(&MultiHeadAttentionConfig::new(
            d_model, n_heads,
        ));
        let input = MhaInput::self_attn(Tensor::random(
            [batch_size, seq_length, d_model],
            Distribution::Standard,
        ));

        let output = mha.forward(input);

        assert_eq!(
            output.context.shape(),
            &Shape::new([batch_size, seq_length, d_model]),
            "Context should have the correct shape",
        );
        assert_eq!(
            output.weights.shape(),
            &Shape::new([batch_size, n_heads, seq_length, seq_length]),
            "Weights should have the correct shape",
        );
    }

    #[test]
    fn test_generic_mha_shapes() {
        let [batch_size, seq_length_1, seq_length_2, d_model, n_heads] = [7, 13, 15, 32, 4];
        let mha = MultiHeadAttention::<TestBackend>::new(&MultiHeadAttentionConfig::new(
            d_model, n_heads,
        ));
        let input = MhaInput::new(
            Tensor::random([batch_size, seq_length_1, d_model], Distribution::Standard),
            Tensor::random([batch_size, seq_length_2, d_model], Distribution::Standard),
            Tensor::random([batch_size, seq_length_2, d_model], Distribution::Standard),
        );

        let output = mha.forward(input);

        assert_eq!(
            output.context.shape(),
            &Shape::new([batch_size, seq_length_1, d_model]),
            "Context should have the correct shape",
        );
        assert_eq!(
            output.weights.shape(),
            &Shape::new([batch_size, n_heads, seq_length_1, seq_length_2]),
            "Weights should have the correct shape",
        );
    }

    #[test]
    pub fn test_self_attention_mask_pad() {
        let [batch_size, seq_length, d_model, n_heads, num_padded] = [3, 6, 32, 2, 2];
        let mha = MultiHeadAttention::new(&MultiHeadAttentionConfig::new(d_model, n_heads));

        // Create a padding mask
        let mask_pad = Tensor::zeros([batch_size, seq_length]);
        let mask_pad = mask_pad.index_assign(
            [0..batch_size, seq_length - num_padded..seq_length],
            &Tensor::ones([batch_size, num_padded]),
        );
        let mask_pad = mask_pad.equal_scalar(1);

        let tensor_1 = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_length, d_model],
            Distribution::Standard,
        );
        // Change the end of the tensor
        let tensor_2 = tensor_1.index_assign(
            [
                0..batch_size,
                seq_length - num_padded..seq_length,
                0..d_model,
            ],
            &Tensor::random([batch_size, num_padded, d_model], Distribution::Standard),
        );

        let input_1 = MhaInput::self_attn(tensor_1).mask_pad(mask_pad.clone());
        let input_2 = MhaInput::self_attn(tensor_2).mask_pad(mask_pad);

        let output_1 = mha.forward(input_1);
        let output_2 = mha.forward(input_2);

        // Check that the begginning of each tensor is the same
        output_1
            .context
            .index([0..batch_size, 0..seq_length - num_padded, 0..d_model])
            .into_data()
            .assert_approx_eq(
                &output_2
                    .context
                    .index([0..batch_size, 0..seq_length - num_padded, 0..d_model])
                    .into_data(),
                3,
            );
    }
}
