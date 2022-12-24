use crate::data::TrainingTextGenerationBatch;
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLoss,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::backend::{ADBackend, Backend},
    tensor::Tensor,
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

    pub fn forward_training(
        &self,
        item: TrainingTextGenerationBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = self.embedding_token.devices()[0];

        let inputs = item.tokens_inputs.to_device(device).detach();
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::<B, 1>::arange_device(0..seq_length, device)
            .reshape([1, seq_length])
            .repeat(0, batch_size)
            .detach();
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn =
            generate_autoregressive_mask::<B>(batch_size, seq_length, embedding.device());

        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn),
        );

        let output = self.output.forward(encoded);
        let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);
        let targets_flatten = item.targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLoss::new(self.vocab_size, Some(self.pad_token));
        let loss = loss.forward(&output_flatten, &targets_flatten);

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }
}

impl<B: ADBackend> TrainStep<B, TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
    for TextClassificationModel<B>
{
    fn step(
        &self,
        item: TrainingTextGenerationBatch<B>,
    ) -> TrainOutput<B, ClassificationOutput<B>> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
    for TextClassificationModel<B>
{
    fn step(&self, item: TrainingTextGenerationBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(item)
    }
}
