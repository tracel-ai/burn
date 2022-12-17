use crate::data::TextGenerationBatch;
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        attention::generate_autoregressive_mask,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::backend::{ADBackend, Backend},
    tensor::{loss::cross_entropy_with_logits, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct TextGenerationModelConfig {
    transformer: TransformerEncoderConfig,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

#[derive(Module, Debug)]
pub struct TextClassificationModel<B: Backend> {
    transformer: Param<TransformerEncoder<B>>,
    embedding_token: Param<Embedding<B>>,
    embedding_pos: Param<Embedding<B>>,
    output: Param<Linear<B>>,
    vocab_size: usize,
    pad_token: usize,
    max_seq_length: usize,
}

impl<B: Backend> TextClassificationModel<B> {
    pub fn new(config: &TextGenerationModelConfig) -> Self {
        let config_embedding_token =
            EmbeddingConfig::new(config.vocab_size, config.transformer.d_model);
        let config_embedding_pos =
            EmbeddingConfig::new(config.max_seq_length, config.transformer.d_model);
        let config_output = LinearConfig::new(config.transformer.d_model, config.vocab_size);

        let transformer = TransformerEncoder::new(&config.transformer);
        let embedding_token = Embedding::new(&config_embedding_token);
        let embedding_pos = Embedding::new(&config_embedding_pos);
        let output = Linear::new(&config_output);

        Self {
            transformer: Param::new(transformer),
            embedding_token: Param::new(embedding_token),
            embedding_pos: Param::new(embedding_pos),
            output: Param::new(output),
            vocab_size: config.vocab_size,
            pad_token: config.pad_token,
            max_seq_length: config.max_seq_length,
        }
    }

    pub fn forward(&self, item: TextGenerationBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens.dims();

        // Teacher forcing
        let inputs = item.tokens.index([0..batch_size, 0..seq_length - 1]);
        let targets = item.tokens.index([0..batch_size, 1..seq_length]);
        let mask_pad = item.mask_pad.index([0..batch_size, 0..seq_length - 1]);

        let seq_length = seq_length - 1;

        let index_positions = Tensor::<B, 1>::arange_device(0..seq_length, inputs.device())
            .reshape([1, seq_length])
            .repeat(0, batch_size);
        let embedding_positions = self.embedding_pos.forward(index_positions.detach());
        let embedding_tokens = self.embedding_token.forward(inputs.detach());
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn =
            generate_autoregressive_mask::<B>(batch_size, seq_length, embedding.device());

        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn),
        );

        let output = self.output.forward(encoded);
        let output_classification = output.reshape([batch_size * seq_length, self.vocab_size]);

        let targets = targets
            .reshape([batch_size * seq_length])
            .to_data()
            .value
            .iter()
            .map(|index| match *index as usize == self.pad_token {
                true => Tensor::<B, 2>::zeros([1, self.vocab_size]),
                false => Tensor::<B, 2>::one_hot(*index as usize, self.vocab_size),
            })
            .collect();

        let targets = Tensor::cat(targets, 0)
            .to_device(output_classification.device())
            .detach();
        let loss = cross_entropy_with_logits(&output_classification, &targets);

        ClassificationOutput {
            loss,
            output: output_classification,
            targets,
        }
    }
}

impl<B: ADBackend> TrainStep<TextGenerationBatch<B>, ClassificationOutput<B>, B::Gradients>
    for TextClassificationModel<B>
{
    fn step(
        &self,
        item: TextGenerationBatch<B>,
    ) -> TrainOutput<ClassificationOutput<B>, B::Gradients> {
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(grads, item)
    }
}

impl<B: Backend> ValidStep<TextGenerationBatch<B>, ClassificationOutput<B>>
    for TextClassificationModel<B>
{
    fn step(&self, item: TextGenerationBatch<B>) -> ClassificationOutput<B> {
        self.forward(item)
    }
}
